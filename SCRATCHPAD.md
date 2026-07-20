# Scratchpad

Your space. Nothing here governs the project, nothing here gets marked, and
you can be wrong in here as often as you like. The formal record lives in
`docs/`; this is for working things out.

Below are small experiments that build the intuition you need for the viva.
All of them run in seconds, on a laptop or a Colab cell, with no graphics card
and no model. Paste one into a cell, run it, and change the numbers until the
behaviour stops surprising you.

Add your own notes underneath each one. Writing the answer in your own words
is the part that makes it stick.

---

## 1. How a reward becomes a nudge

The training method never asks "is this good?". It only asks "is this better
than the other attempts at the same question?". That comparison is the whole
mechanism.

```python
import numpy as np

def advantages(scores):
    """What GRPO actually computes. Positive means push toward it."""
    scores = np.array(scores, dtype=float)
    return scores - scores.mean()

print(advantages([1, 0, 0, 1, 0, 1, 0, 0]))
```

Run it. Three attempts got the answer right, five didn't. The right ones come
out positive, the wrong ones negative. The model is pushed toward the positive
ones.

**Now try this:**

```python
print(advantages([0, 0, 0, 0, 0, 0, 0, 0]))   # all wrong
print(advantages([1, 1, 1, 1, 1, 1, 1, 1]))   # all right
```

Both give all zeros. **When every attempt scores the same, the model learns
nothing from that question.** Not a little, nothing. This is the single most
important mechanical fact about the method you are using.

> Your notes:

---

## 2. Why your measurements had to be fractions

This is exercise 1 applied to your actual design decision.

```python
import random

def ties(reward_fn, trials=1000, group=8):
    """How often does a whole group score identically?"""
    n = 0
    for _ in range(trials):
        scores = [reward_fn() for _ in range(group)]
        if len(set(scores)) == 1:
            n += 1
    return n / trials

# A yes/no reward: right or wrong
binary = lambda: random.choice([0, 1])

# A continuous reward: correctness plus a structure score
continuous = lambda: random.choice([0, 1]) + 0.5 * round(random.random(), 3)

print("binary tie rate:    ", ties(binary))
print("continuous tie rate:", ties(continuous))
```

The binary reward wastes a real fraction of every training run. The continuous
one almost never does.

This is why every function in `reward/features.py` returns a fraction rather
than a yes/no. It was not a style choice. A yes/no reward would have failed
quietly, in a way that looks exactly like "the idea didn't work".

> Your notes:

---

## 3. Score a trace by hand

Do this once before trusting any code, including mine.

```python
from reward.xaif_build import build_trace
from reward.features import compute
from reward.ari import Relation, RelationResult

trace = build_trace(
    "All ravens are black.\n"
    "Every bird in this garden is a raven.\n"
    "Therefore every bird in this garden is black.\n"
    "The answer is black."
)
print("steps:      ", trace.steps)
print("conclusion: ", trace.conclusion, "at index", trace.conclusion_index)

# Hand-built relations: steps 0 and 1 both support step 2.
rels = RelationResult(relations=[
    Relation(source=0, target=2, kind="RA", confidence=0.95),
    Relation(source=1, target=2, kind="RA", confidence=0.95),
])

for name, value in compute(trace, rels).as_dict().items():
    print(f"{name:20s} {value:.3f}")
```

**Before you run it, predict each of the six numbers.** Write them down. Then
compare. Where you were wrong is where your mental model of your own reward is
wrong, and that is exactly what a viva finds.

> Your predictions:
> Your notes on where you were wrong:

---

## 4. Try to cheat your own reward

This one is an exercise and a research contribution at the same time. Anything
you find here goes into the reward-hacking chapter.

Write a trace that scores well but reasons badly. Ideas to start from:

- Say the same thing three times in different words
- Split one claim into six short fragments
- Stuff it with "therefore", "because", "it follows that"
- Assert a conclusion repeatedly without ever supporting it

```python
padded = build_trace(
    "The birds here are ravens.\n"
    "The birds in this garden are ravens.\n"          # restatement
    "It is the case that these birds are ravens.\n"   # restatement again
    "Therefore they are black.\n"
    "The answer is black."
)
# score it and compare against the clean version above
```

Every trick that raises the score is a hole in the reward. Every trick that
fails to raise it is evidence the design resists that exploit. **Both outcomes
are results.** Keep a list here as you find them.

> Exploits that worked:
> Exploits that failed (and why the design stopped them):

---

## 5. Watch the leash

`kl_coef` in your configs penalises the model for drifting too far from where
it started. Without it, models find degenerate text that scores well and stop
producing language.

```python
def total(reward, drift, kl_coef=0.04):
    return reward - kl_coef * drift

for drift in [0, 1, 5, 20, 100]:
    print(f"drift {drift:4d} -> {total(1.0, drift):.2f}")
```

Your ablation condition, where the structure score is the only reward and
correctness is dropped, is a deliberate test of what happens when this pressure
goes unchecked. Predict what those traces will look like before you run it, and
write the prediction here. Comparing it to what actually happened is a strong
paragraph in the discussion.

> My prediction for what condition D traces will look like:

---

## 6. Questions to be able to answer without notes

Come back and fill these in as you learn them. These are the ones a marker
probes.

- Why does this method not need a second model to judge how good a state is?
- What happens to learning when all attempts in a group score the same, and why
  does that constrain how the reward must be designed?
- Why is trace length included as a covariate in the correlation study?
- Why can't the confidence thresholds in the relation model be tuned?
- Why are extraction failures counted separately from wrong answers?
- Why does better-structured reasoning not guarantee more honest reasoning?

> Answers, in my own words:

---

## Running notes

Date them. Rough is fine.
