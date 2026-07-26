# 01 How the repo fits together

The whole project is one idea: **score a model's reasoning by turning it into a
diagram and measuring the diagram, then use that score to train the model.**

Every file exists to serve one of three jobs:

1. **Score a piece of reasoning** (the `reward/` folder). This is the core.
2. **Train a model using that score** (the `train/` folder, not built yet).
3. **Measure whether it worked** (`eval/` and `analysis/`, not built yet).

Everything else is support: data loading, configuration, tests, docs.

---

## The one path everything flows along

Follow a single reasoning trace through the code and you understand the repo.

```
A model writes a chain of thought:
  "All ravens are black. Every bird here is a raven.
   Therefore every bird here is black. The answer is black."
         |
         |   reward/xaif_build.py
         v
Split into numbered steps, find the conclusion:
  0: All ravens are black.
  1: Every bird here is a raven.
  2: Therefore every bird here is black.   <- conclusion
         |
         |   reward/ari.py   (this is the model Ramon's group trained)
         v
Ask, for each PAIR of steps, how they relate:
  step 0 -> step 2 : supports
  step 1 -> step 2 : supports
         |
         |   reward/features.py
         v
Build a diagram (steps are dots, relations are arrows) and measure it:
  how much connects to the conclusion?   1.0
  how long is the reasoning chain?        ...
  how much is just repetition?            0.0
  (six numbers, each between 0 and 1)
         |
         |   reward/scorer.py
         v
Combine into ONE number = the reward:
  reward = (was the answer correct?) + 0.5 x (how good was the structure?)
         |
         v
That number goes back to the training loop, which nudges the model
toward reasoning that scored well.
```

If you understand that column, you understand the project. Everything below is
just detail about each step.

---

## The files, grouped by job

### Scoring (the `reward/` folder) — BUILT

| File | Plain-language job | Check it yourself |
|---|---|---|
| `xaif_build.py` | Cut a trace into numbered steps; find the conclusion; flag traces too small to bother scoring | run exercise 3 in `SCRATCHPAD.md` |
| `ari.py` | The relation model. Given two steps, say whether one supports, contradicts, or restates the other | `tests/test_reward.py`, the pairing tests |
| `features.py` | Turn the diagram into six measurements, each a fraction | the feature tests |
| `scorer.py` | Combine the six into one reward number; refuse to run if the weights were never set properly | the scorer tests |
| `calibration.yaml` | Where the weights and thresholds live. Empty on purpose until Week 2 | — |

### Data and support — BUILT

| File | Plain-language job |
|---|---|
| `data.py` | Load the questions. Split them into the five tiers (train, held-out, transfer, LogiQA, GSM8K). Refuse to run if a tier overlaps with training |
| `docs/splits.json` | The frozen decision about which question families are trained on and which are held back |
| `utils.py` | Crash-safe saving, so a dead Colab session costs you one item, not a day |
| `configs/*.yaml` | The three experiment settings: baseline, structural, ablation |

### Training and evaluation — NOT BUILT YET

| File | Will do | Why it is not built yet |
|---|---|---|
| `train/grpo_train.py` | The actual training loop | Depends on which training library version Colab gives you; writing it blind produces a script that fails on line 40 |
| `eval/bbh_eval.py` | Accuracy across the five tiers | Needs a trained model to evaluate |
| `analysis/c1_correlation.py` | The correlation study (your insurance policy) | Needs the baseline corpus first |
| `calibrate.py` | Sets the weights in `calibration.yaml` from the correlation study | Needs the corpus and the study |

### The record — BUILT

| File | Job |
|---|---|
| `docs/experiment_plan.md` | The pre-registration: what you will do, decided before seeing data |
| `docs/deviation_log.md` | Every change from that plan, dated, with a reason |
| `docs/run_log.md` | Every training run, with its commit reference so it is reproducible |
| `tests/test_reward.py` | Proof the scoring behaves, including that padding cannot raise the score |

---

## Why it is split this way, not all in one file

Two reasons, both practical.

**You can test the scoring without a graphics card.** The heavy model imports
live inside the one function that needs them, so everything else runs on your
laptop in under a second. You develop and test the logic locally, and only
touch Colab when you actually need the GPU.

**Each piece can be checked alone.** If a reward number looks wrong, you can
feed a hand-built diagram straight into `features.py` and see if the
measurement is wrong, without running a model at all. That is what the tests
do, and it is why a bug has nowhere to hide.

---

## The one rule that holds the whole thing together

**The logic lives in these `.py` files, never in notebook cells.** When you
work on Colab, the cell you run is one or two lines that call into these files.
The reason is that you will run five training runs, and if the scoring code
lived in cells you would end up with five slightly different copies and no way
to know which one produced which result. Files are version-controlled; cells
drift.

---

## Question to answer in your own words

If a trained model starts scoring well by repeating the same reasoning step in
slightly different words, which of the six measurements should catch it, and
which file would you open to check that it does?

> Your answer:

(If you are not sure, this is exactly exercise 4 in `SCRATCHPAD.md`.)
