"""
Train the model with GRPO.

GRPO is the training method. For each question the model writes several answers,
each answer gets a score, and the model is pushed towards the higher scoring
ones. If all the answers for a question get the same score, that question
teaches the model nothing.

Step 2 runs this with a FAKE reward (just a random number). The only goal is to
check that the whole training loop runs on Colab without crashing. The real
argument structure reward is added later.

How to run it from a notebook cell:
    from train.grpo_train import run
    run("configs/baseline.yaml", max_steps=5)
"""

import os

# Turn off torch.compile. Unsloth's GRPO trainer tries to compile part of the
# training step and it crashes on this model. Running it plain (eager) is
# slower but it works. This must be set before unsloth is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import random

import yaml
from datasets import Dataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_prompt(question):
    # Ask the model to reason step by step and end in a fixed format, so later
    # we can read the answer back out with a simple rule.
    return (
        "Answer the question. Think step by step. "
        "Finish with a line that says 'The answer is X'.\n\n"
        "Question: " + question + "\n"
    )


def dummy_dataset():
    # A few simple questions for the smoke test. Using these instead of the real
    # data keeps the test focused on one thing: does the training loop run.
    questions = [
        "If all cats are animals and Tom is a cat, is Tom an animal?",
        "Sarah is taller than Jane. Jane is taller than Amy. Who is tallest?",
        "There are 3 red balls and 2 blue balls. How many balls are there?",
        "If it rains the ground gets wet. It is raining. Is the ground wet?",
    ]
    rows = [{"prompt": make_prompt(q)} for q in questions]
    return Dataset.from_list(rows)


def fake_reward(prompts, completions, **kwargs):
    # One random score per answer. Only here to prove the loop turns over.
    return [random.random() for _ in completions]


def run(config_path, max_steps=5, num_generations=None, beta=0.0):
    # beta is the KL penalty strength. We set it to 0 for the smoke test, which
    # skips the reference-model step (that step is where the compile crash
    # happened). Real training will turn KL back on once the loop is proven.
    # unsloth and trl are imported inside the function, not at the top, so this
    # file can still be imported on a laptop with no GPU. They only load when
    # you actually train.
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    cfg = load_config(config_path)
    g = cfg["grpo"]
    n_gen = num_generations or g["num_generations"]

    # Load the model in 4-bit so it fits on a small GPU.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=1024,
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    # Add the small trainable adapters (LoRA). We only train these, not the
    # whole model, which is what makes it cheap.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
    )

    data = dummy_dataset()

    args = GRPOConfig(
        output_dir="outputs/" + cfg["run_id"],
        num_generations=n_gen,
        per_device_train_batch_size=n_gen,
        learning_rate=g["learning_rate"],
        max_steps=max_steps,
        max_completion_length=g["max_completion_length"],
        beta=beta,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=fake_reward,
        args=args,
        train_dataset=data,
    )
    trainer.train()
    print("training loop finished without crashing")
    return trainer
