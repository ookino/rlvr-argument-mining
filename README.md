# Argument-Structure Rewards for Reasoning Models

MSc dissertation code. University of Dundee, supervised by Dr. Ramon Ruiz-Dolz.

## What this does, in plain terms

When you train an AI to reason by rewarding it, the reward normally depends
only on whether the final answer was right. Nothing checks the reasoning in
between, so a model can reason badly, get lucky, and be rewarded for it.

This project scores the reasoning itself. It splits a model's chain of thought
into steps, uses an argument-mining model to work out which steps support,
contradict, or restate which others, and measures properties of the resulting
diagram: how much of the reasoning connects to the conclusion, how long the
supporting chains are, how much is redundant padding. Those measurements
become part of the training reward.

The difference from existing work is that the score comes from a recovered
diagram and ordinary arithmetic, not from a second AI grading the reasoning.
It is transparent, deterministic, and needs no training data of its own.

## Layout

```
reward/     the scoring pipeline
  xaif_build.py   trace -> reasoning steps -> graph nodes
  ari.py          the relation model: which steps support/contradict/restate which
  features.py     graph -> six measurements
  scorer.py       measurements -> one number
  calibration.yaml   frozen weights and thresholds (week 2)
train/      the training script
eval/       accuracy on held-out and transfer question families
analysis/   the correlation study and reward-hacking inspection
configs/    baseline, structural, ablation
notebooks/  Colab entry points; these call the code, they do not contain it
docs/       experiment plan, run log, deviation log, data splits
tests/      run these before trusting anything
```

## Setup

Laptop (scoring logic, tests, analysis; no graphics card needed):

```
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

Colab: paste `notebooks/00_bootstrap.py` as the first cell of any notebook.

## The three experiments

| Config | What it is |
|---|---|
| `baseline.yaml` | Reward on correct answers only. The comparison. |
| `structural.yaml` | Correctness plus the argument-structure score. The thesis. |
| `ablation.yaml` | Structure score minus the contradiction and restatement parts. Tests whether argument content matters or only signal density. |

## Two things that are deliberate

**The scoring weights refuse to be set by hand.** They come from the
correlation study in week 2, and `reward/scorer.py` raises an error if they are
missing. Hand-tuning a reward and then reporting an improvement is the thing
that makes a result worthless.

**One setting dominates the compute budget.** `ari.window` decides whether
every reasoning step is compared to every other (thorough, grows with the
square of trace length) or only to its neighbours (cheap, misses long-range
links). For a 15-step trace that is 105 comparisons versus 14. Measure both
before choosing.

## Reproducing the headline result

```
bash scripts/reproduce.sh
```

Every run appends its commit reference, config, seed, and result to
`docs/run_log.md`. Every departure from `docs/experiment_plan.md` is recorded
in `docs/deviation_log.md` on the day it happens.
