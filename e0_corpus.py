"""
Build the E0 corpus: for each question, let the base model reason, check if
it's right, and score its argument structure. One line per question.

    from e0_corpus import generate
    generate(n_questions=200)
"""

import re

import yaml

from data import load_splits
from reward.features import compute
from reward.xaif_build import build_trace, extract_answer
from utils import resumable

# A real reasoning model (thinks inside <think>...</think>), per the supervisor.
# 4B for the corpus; swap to Qwen/Qwen3-1.7B for the lighter training runs.
# The old Qwen/Qwen2.5-3B-Instruct corpus is kept as the instruct-model baseline.
MODEL = "Qwen/Qwen3-4B"


# The expected answer format for each task family. Without this the model
# answers in the task's own words ("plausible" instead of "yes", "T" instead of
# "Yes"), which we then wrongly mark incorrect. Telling it the format up front
# is standard practice for evaluating on these benchmarks.
ANSWER_FORMATS = {
    "web_of_lies": "Yes or No",
    "sports_understanding": "yes or no",
    "navigate": "Yes or No",
    "formal_fallacies": "valid or invalid",
    "disambiguation_qa": "the letter of the correct option, for example (A)",
    "logical_deduction_three_objects": "the letter of the correct option, for example (A)",
    "logical_deduction_five_objects": "the letter of the correct option, for example (A)",
    "logical_deduction_seven_objects": "the letter of the correct option, for example (A)",
}


def build_prompt(question, family=None):
    # Ask for step by step reasoning and, importantly, tell the model exactly
    # what the final answer should look like for this task. No "X" placeholder:
    # the small model took that literally and wrote "The answer is X".
    lines = [
        "Answer the question below. Think step by step, one step per line.",
        "Then write your final answer on the last line, after the words "
        "'The answer is'.",
    ]
    fmt = ANSWER_FORMATS.get(family)
    if fmt:
        lines.append(f"Your final answer must be {fmt}.")
    lines += ["", "Question: " + question]
    return "\n".join(lines)


def normalise(text):
    # Lowercase and strip brackets, punctuation, and markdown, so "(A)", "a.",
    # and "**A**" all compare equal.
    return re.sub(r"[*#`()\[\].,:]", "", text.strip().lower()).strip()


# Words that mean the same answer. The model says "plausible" or "true" when the
# gold answer is "yes"; these map them together so we do not mark a right answer
# wrong.
_SYNONYMS = {
    "plausible": "yes", "true": "yes", "t": "yes",
    "implausible": "no", "false": "no", "f": "no",
}


def _canon(word):
    return _SYNONYMS.get(word, word)


def is_correct(predicted, gold):
    if predicted is None:
        return False
    p, g = normalise(predicted), normalise(gold)
    if not p or not g:
        return False
    # Map synonyms so "plausible" counts as "yes", etc.
    if _canon(p) == _canon(g):
        return True
    if p == g:
        return True
    # The gold answer must appear as a WHOLE WORD in the prediction. This stops
    # "valid" from matching inside "invalid".
    return re.search(rf"\b{re.escape(g)}\b", p) is not None


def load_generator(model_name=MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    return model, tokenizer


def generate_trace(model, tokenizer, question, family=None,
                   max_new_tokens=4096, temperature=0.6):
    # NOTE: reasoning models think for a long time before answering, so
    # max_new_tokens is much higher than the ~512 an instruct model needed. If a
    # trace has no answer after </think>, it was likely cut off - raise this.
    # Sampling settings follow the Qwen3 model card for thinking mode
    # (temp 0.6, top_p 0.95, top_k 20); they warn against greedy decoding.
    import torch

    messages = [{"role": "user", "content": build_prompt(question, family)}]
    # enable_thinking defaults to True for Qwen3, so the <think> block comes for
    # free - no need to pass it. (A non-Qwen3 tokenizer just ignores the default.)
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Keep only the newly generated part, not the prompt. Decode WITHOUT skipping
    # special tokens, because Qwen3's <think>/</think> are special tokens and
    # skip_special_tokens=True would strip the very markers we split on. Remove
    # the chat control tokens by hand instead.
    new_tokens = out[0][enc.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    for control in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(control, "")
    return text.strip()


def preview(n=3, max_new_tokens=4096, family=None, seed=13):
    # Generate a few traces and print the raw output plus what build_trace pulls
    # out of it. For eyeballing a new model before the full run. Saves nothing.
    # Returns the raw texts so you can poke at them in the notebook.
    import random

    splits = load_splits(include_external=False)
    questions = list(splits.train)
    random.Random(seed).shuffle(questions)
    if family:
        questions = [q for q in questions if q.family == family]
    questions = questions[:n]

    model, tokenizer = load_generator()
    raw_texts = []
    for i, item in enumerate(questions, 1):
        text = generate_trace(model, tokenizer, item.question, item.family, max_new_tokens)
        raw_texts.append(text)
        trace = build_trace(text)
        answer, _ = extract_answer(text)
        has_think = "<think>" in text.lower()
        print("=" * 70)
        print(f"[{i}] {item.family}  (gold: {item.answer})")
        print(f"has <think> tag: {has_think} | steps: {trace.n_steps} | "
              f"answer: {answer} | correct: {is_correct(answer, item.answer)}")
        print("--- raw output (first 2000 chars) ---")
        print(text[:2000])
        print("--- parsed steps ---")
        for j, s in enumerate(trace.steps):
            print(f"  {j}. {s}")
        print()
    return raw_texts


def score_trace(ari, trace_text, window=None, dedupe=False):
    # Split into steps, find the relations, measure the graph. The pipeline
    # already returns zeros for traces too small to have any structure.
    # dedupe=True is Arm B: collapse the model's repetition first (D-010).
    trace = build_trace(trace_text, dedupe=dedupe)
    relations = ari.identify(trace.steps, window=window)
    features = compute(trace, relations)
    return trace, relations, features


def generate(
    n_questions=200,
    out="results/E0/corpus.jsonl",
    window=None,
    max_new_tokens=4096,      # reasoning models think long; see generate_trace
    fresh=False,
):
    import random

    from reward.ari import ARI

    # fresh=True wipes the old file first, so a regeneration cannot silently
    # skip everything as "already done". Use it whenever the code or the
    # sampling changed and you want a clean corpus.
    if fresh:
        reset_corpus(out)

    # Real training questions. Shuffle first (fixed seed, so it is reproducible)
    # so the sample is spread across all the training families. Without this,
    # taking the first N gives questions from only the first family.
    splits = load_splits(include_external=False)
    questions = list(splits.train)
    random.Random(13).shuffle(questions)
    questions = questions[:n_questions]

    families = sorted({q.family for q in questions})
    print(f"scoring {len(questions)} questions across {len(families)} families")

    model, tokenizer = load_generator()
    ari = ARI()

    def work(item):
        trace_text = generate_trace(model, tokenizer, item.question,
                                    item.family, max_new_tokens)
        answer, extracted = extract_answer(trace_text)
        trace, relations, features = score_trace(ari, trace_text, window)
        return {
            "id": item.id,
            "family": item.family,
            "gold": item.answer,
            "answer": answer,
            "extract_ok": extracted,
            "correct": is_correct(answer, item.answer),
            "n_steps": trace.n_steps,
            "n_relations": len(relations.relations),
            "conclusion_index": trace.conclusion_index,
            "relations": _relations_to_dicts(relations),
            "trace": trace_text,
            **features.as_dict(),
        }

    n_new = resumable(questions, out, work, id_fn=lambda it: it.id)
    print(f"done. {n_new} new traces written to {out}")
    summarise(out)


def _relations_to_dicts(relations):
    # Save the raw links so features can be recomputed later without re-running
    # ARI. s=source step, t=target step, k=kind (RA/CA/MA), c=confidence.
    return [{"s": r.source, "t": r.target, "k": r.kind, "c": round(r.confidence, 3)}
            for r in relations.relations]


def rescore(in_path="results/E0/corpus.jsonl",
            out_path="results/E0/corpus_rescored.jsonl", window=None, dedupe=False):
    """Re-score an existing corpus with the CURRENT pipeline, without
    regenerating the traces. Runs ARI on the stored trace text (fast: no model
    generation) and recomputes the features. Use this after changing the
    splitting, the conclusion detection, or the feature code. Saves the raw
    relations too, so the NEXT change needs no ARI at all (see rescore_local).
    dedupe=True re-scores under Arm B (repetition collapsed) - run it into a
    second file and compare summarise() of both to see the effect (D-010)."""
    from reward.ari import ARI
    from utils import read_jsonl

    ari = ARI()
    rows = list(read_jsonl(in_path))
    print(f"re-scoring {len(rows)} traces from {in_path} (dedupe={dedupe})")

    def work(row):
        trace, relations, features = score_trace(ari, row["trace"], window, dedupe=dedupe)
        keep = {k: row[k] for k in
                ("id", "family", "gold", "answer", "extract_ok", "correct", "trace")
                if k in row}
        return {
            **keep,
            "n_steps": trace.n_steps,
            "n_relations": len(relations.relations),
            "conclusion_index": trace.conclusion_index,
            "relations": _relations_to_dicts(relations),
            **features.as_dict(),
        }

    n = resumable(rows, out_path, work, id_fn=lambda r: str(r["id"]))
    print(f"done. {n} traces re-scored to {out_path}")
    summarise(out_path)


def summarise(out="results/E0/corpus.jsonl"):
    # A quick look at what we made: how many correct, average structure scores.
    from utils import read_jsonl

    rows = list(read_jsonl(out))
    if not rows:
        print("no rows yet")
        return

    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    extract_fail = sum(1 for r in rows if not r.get("extract_ok"))
    print(f"\n{n} traces")
    print(f"correct:            {correct}/{n}  ({100*correct/n:.0f}%)")
    print(f"extraction failed:  {extract_fail}/{n}")
    print(f"mean steps:         {sum(r['n_steps'] for r in rows)/n:.1f}")
    for feat in ["support_density", "attachment", "connectivity",
                 "support_depth", "conflict_rate", "restatement_rate"]:
        vals = [r[feat] for r in rows if feat in r]
        if vals:
            print(f"mean {feat:18s} {sum(vals)/len(vals):.3f}")

    # Per family: we want a spread, and enough wrong answers in each to study.
    from collections import defaultdict
    fam = defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += bool(r.get("correct"))
    print("\nby family:")
    for f, (fn, fc) in sorted(fam.items()):
        print(f"  {f:34s} {fc:3d}/{fn:<3d} correct ({100*fc//max(fn,1):2d}%)")

    # Early directional hint for the correlation (RQ2): do correct traces look
    # structurally different from wrong ones? This is a SMELL TEST on a small
    # run, not a result - too few traces for a real AUC. But if the columns
    # differ in the expected direction, the eventual AUC is likely > 0.5.
    feats = ["support_density", "attachment", "connectivity",
             "support_depth", "conflict_rate", "restatement_rate"]
    yes = [r for r in rows if r.get("correct")]
    no = [r for r in rows if not r.get("correct")]
    if yes and no:
        print(f"\nstructure means, correct ({len(yes)}) vs wrong ({len(no)}):")
        print(f"  {'feature':18s} {'correct':>8s} {'wrong':>8s} {'diff':>8s}")
        for feat in feats:
            cy = [r[feat] for r in yes if feat in r]
            cn = [r[feat] for r in no if feat in r]
            if cy and cn:
                my, mn = sum(cy) / len(cy), sum(cn) / len(cn)
                print(f"  {feat:18s} {my:8.3f} {mn:8.3f} {my - mn:+8.3f}")
        print("  want: support measures correct>wrong (+); "
              "conflict/restatement correct<wrong (-)")
    else:
        print("\n(need both correct and wrong traces for the correct-vs-wrong split)")


def reset_corpus(out="results/E0/corpus.jsonl"):
    # Delete the corpus file so a regeneration starts completely fresh. Run this
    # before regenerating if the sampling or labelling has changed, otherwise
    # the resumable writer keeps the old rows.
    import os

    if os.path.exists(out):
        os.remove(out)
        print("deleted old corpus:", out)
    else:
        print("no corpus to delete at", out)


def inspect(out="results/E0/corpus.jsonl", only="all", family=None, n=5):
    # Print a few full traces so we can eyeball whether the correctness labels
    # are trustworthy. only can be "all", "correct", "wrong", or "extract_fail".
    # family filters to one task family, e.g. "sports_understanding".
    from utils import read_jsonl

    rows = list(read_jsonl(out))
    shown = 0
    for r in rows:
        if family and r["family"] != family:
            continue
        if only == "correct" and not r["correct"]:
            continue
        if only == "wrong" and r["correct"]:
            continue
        if only == "extract_fail" and r["extract_ok"]:
            continue
        print(f"\n[{r['family']}]  gold={r['gold']!r}  answer={r['answer']!r}  "
              f"correct={r['correct']}  extract_ok={r['extract_ok']}")
        print("trace (last 300 chars):", r["trace"][-300:])
        shown += 1
        if shown >= n:
            break


def generate_probe(n_questions=20, k=8, out="results/E0/probe.jsonl",
                   window=10, max_new_tokens=4096, fresh=False, dedupe=False):
    # Reward-design variance probe: k completions per question, so within-group
    # (per-question) reward variance can be measured - the thing GRPO actually
    # needs. Each row carries question_id and completion so groups can be formed.
    import random
    from reward.ari import ARI

    if fresh:
        reset_corpus(out)
    splits = load_splits(include_external=False)
    questions = list(splits.train)
    random.Random(13).shuffle(questions)
    questions = questions[:n_questions]
    jobs = [(item, j) for item in questions for j in range(k)]
    print(f"{len(questions)} questions x {k} completions = {len(jobs)} traces")

    model, tokenizer = load_generator()
    ari = ARI()

    def work(job):
        item, j = job
        trace_text = generate_trace(model, tokenizer, item.question, item.family, max_new_tokens)
        answer, extracted = extract_answer(trace_text)
        trace, relations, features = score_trace(ari, trace_text, window, dedupe=dedupe)
        return {
            "id": f"{item.id}#{j}",
            "question_id": item.id,
            "completion": j,
            "family": item.family,
            "gold": item.answer,
            "answer": answer,
            "extract_ok": extracted,
            "correct": is_correct(answer, item.answer),
            "n_steps": trace.n_steps,
            "n_relations": len(relations.relations),
            "trace": trace_text,
            **features.as_dict(),
        }

    n_new = resumable(jobs, out, work, id_fn=lambda job: f"{job[0].id}#{job[1]}")
    print(f"done. {n_new} completions written to {out}")


def generate_probe_batched(n_questions=20, k=8, out="results/E0/probe.jsonl",
                           window=10, max_new_tokens=4096, batch_size=16):
    # Batched version of the probe generator: generates a whole batch of prompts
    # per forward pass instead of one at a time (~10x faster on an A100).
    # Resumable: skips ids already in `out`. do_sample makes each row differ, so
    # the k completions of a question are independent samples.
    import os, json as _json, random, torch
    from reward.ari import ARI

    splits = load_splits(include_external=False)
    questions = list(splits.train)
    random.Random(13).shuffle(questions)
    questions = questions[:n_questions]
    jobs = [(item, j) for item in questions for j in range(k)]

    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(_json.loads(line)["id"])
            except Exception:
                pass
    jobs = [job for job in jobs if f"{job[0].id}#{job[1]}" not in done]
    print(f"{len(jobs)} completions to generate ({len(done)} already done), batch {batch_size}")

    model, tokenizer = load_generator()
    tokenizer.padding_side = "left"   # required for correct batched decoder generation
    ari = ARI()

    with open(out, "a") as fh:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_prompt(it.question, it.family)}],
                    add_generation_prompt=True, tokenize=False)
                for it, _ in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.6, top_p=0.95, top_k=20,
                    pad_token_id=tokenizer.pad_token_id)
            gen = out_ids[:, enc.input_ids.shape[1]:]
            for (it, j), row in zip(batch, gen):
                text = tokenizer.decode(row, skip_special_tokens=False)
                for c in ("<|im_end|>", "<|endoftext|>"):
                    text = text.replace(c, "")
                text = text.strip()
                answer, extracted = extract_answer(text)
                trace, relations, features = score_trace(ari, text, window)
                fh.write(_json.dumps({
                    "id": f"{it.id}#{j}", "question_id": it.id, "completion": j,
                    "family": it.family, "gold": it.answer, "answer": answer,
                    "extract_ok": extracted, "correct": is_correct(answer, it.answer),
                    "n_steps": trace.n_steps, "n_relations": len(relations.relations),
                    "trace": text, **features.as_dict()}) + "\n")
            fh.flush()
            print(f"  {min(start + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    print("done ->", out)
