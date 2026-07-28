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


def build_prompt(question):
    # Ask for step by step reasoning and a fixed final line, so we can read the
    # answer back out with a simple rule.
    return (
        "Answer the question. Think step by step, one step per line. "
        "Then finish with a line that says exactly 'The answer is X'.\n\n"
        "Question: " + question
    )


def normalise(text):
    # Lowercase, strip spaces and brackets and full stops, so "(A)" and "a."
    # and "A" all compare equal.
    return re.sub(r"[()\[\].,]", "", text.strip().lower()).strip()


def is_correct(predicted, gold):
    if predicted is None:
        return False
    p, g = normalise(predicted), normalise(gold)
    if not p or not g:
        return False
    # Exact match, or one contains the other (handles "yes" vs "the answer is yes").
    return p == g or g in p or p in g


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


def generate_trace(model, tokenizer, question, max_new_tokens=512, temperature=0.7):
    import torch

    messages = [{"role": "user", "content": build_prompt(question)}]
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
):
    from reward.ari import ARI

    # Real training questions, in a fixed order so this is reproducible.
    splits = load_splits(include_external=False)
    questions = splits.train[:n_questions]
    print(f"scoring {len(questions)} questions from the training split")

    model, tokenizer = load_generator()
    ari = ARI()

    def work(item):
        trace_text = generate_trace(model, tokenizer, item.question, max_new_tokens)
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
