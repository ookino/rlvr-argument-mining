"""
Train the model with GRPO.

GRPO is the training method. For each question the model writes several answers,
each answer gets a score, and the model is pushed towards the higher scoring
ones. If all the answers for a question get the same score, that question
teaches the model nothing.

Step 2 runs this with a FAKE reward (just a random number). The only goal is to
check the whole training loop runs without crashing. The real argument structure
reward is added later.

We use plain TRL here, not Unsloth. Unsloth is a speed add-on, but its current
version has a bug in its GRPO trainer (an off-by-one in the log-prob step). TRL
is the standard library underneath and it works. For real training we get the
speed back with vLLM, which plugs into TRL. See docs/deviation_log.md D-007.

How to run it from a notebook cell:
    from train.grpo_train import run
    run("configs/baseline.yaml", max_steps=5)
"""

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


def run(config_path, max_steps=5, num_generations=None, beta=0.0, max_completion=128):
    # Imported inside the function so this file still imports on a laptop with
    # no GPU. The heavy libraries only load when you actually train.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig

    # TRL eagerly loads an optional model-merging helper called mergekit. On
    # this runtime mergekit is half-installed: TRL believes it is available but
    # importing it fails. We do not use model merging, so we tell TRL it is not
    # available before importing the trainer. That makes TRL skip the broken
    # import. This is done here in code so it works on any runtime, whatever
    # leftover packages are lying around.
    import trl.import_utils as _trl_import_utils
    _trl_import_utils.is_mergekit_available = lambda: False

    from trl import GRPOConfig, GRPOTrainer

    cfg = load_config(config_path)
    g = cfg["grpo"]
    n_gen = num_generations or g["num_generations"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model in 4-bit so it fits on a small GPU.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=bnb,
        device_map="auto",
    )

    # The small trainable adapters (LoRA). We only train these, which is what
    # makes it cheap. TRL applies them to the 4-bit model for us.
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    data = dummy_dataset()

    args = GRPOConfig(
        output_dir="outputs/" + cfg["run_id"],
        num_generations=n_gen,
        per_device_train_batch_size=n_gen,
        learning_rate=g["learning_rate"],
        max_steps=max_steps,
        max_completion_length=max_completion,
        beta=beta,               # 0 for the smoke test: skips the KL step
        bf16=True,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=fake_reward,
        args=args,
        train_dataset=data,
        peft_config=lora,
    )
    trainer.train()
    print("training loop finished without crashing")
    return trainer
