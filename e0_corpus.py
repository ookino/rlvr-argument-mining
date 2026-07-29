"""
Make the baseline trace corpus (experiment E0).

For each training question we do four things:
  1. ask the base model to reason step by step and give an answer
  2. check if the answer is correct
  3. score the reasoning with the argument pipeline (split into steps, find the
     relations between them, measure the resulting graph)
  4. save everything to one file, one line per question

This is done with the untrained base model, so it tells us how the model
reasons before any training. It is the input to the correlation study: do the
argument scores line up with getting the answer right?

Run it from a notebook:
    from e0_corpus import generate
    generate(n_questions=200)
"""

import re

import yaml

from data import load_splits
from reward.features import compute
from reward.xaif_build import build_trace, extract_answer
from utils import resumable

MODEL = "Qwen/Qwen2.5-3B-Instruct"


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
                   max_new_tokens=512, temperature=0.7):
    import torch

    messages = [{"role": "user", "content": build_prompt(question, family)}]
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
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Keep only the newly generated part, not the prompt.
    new_tokens = out[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_trace(ari, trace_text, window=None):
    # Split into steps, find the relations, measure the graph. The pipeline
    # already returns zeros for traces too small to have any structure.
    trace = build_trace(trace_text)
    relations = ari.identify(trace.steps, window=window)
    features = compute(trace, relations)
    return trace, relations, features


def generate(
    n_questions=200,
    out="results/E0/corpus.jsonl",
    window=None,
    max_new_tokens=512,
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
            "trace": trace_text,
            **features.as_dict(),
        }

    n_new = resumable(questions, out, work, id_fn=lambda it: it.id)
    print(f"done. {n_new} new traces written to {out}")
    summarise(out)


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
