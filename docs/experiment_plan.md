# Experiment Plan

**Structured Argumentative Reward Signals for Process-Level Reinforcement Learning in Reasoning Language Models**

MSc Artificial Intelligence, University of Dundee. Supervisor: Dr. Ramon Ruiz-Dolz.

This document is a **pre-registration**. Every analysis below is specified before the data is seen, so that no result can be the product of choices made after looking at it. It supersedes `experiment_plan_v1_archived.md`, which was written for a twelve-week schedule; the reduction to four weeks is recorded as D-003 in `deviation_log.md`.

Any departure from this document is logged in `deviation_log.md` on the day it happens, with a reason. That log is part of the submission.

---

## 1. The claim

When a language model is trained by reinforcement learning to reason, the reward almost always depends only on whether the final answer is correct. The reasoning in between is unconstrained. A model can therefore reason badly and be rewarded for it, provided it arrives somewhere right.

The field's response has been process-level rewards, which score the reasoning itself. Every existing approach derives that score from a learned model or a language model acting as a judge: a process reward model, a rubric scored by a large model, a co-trained step discriminator. All of them are opaque, need training data, and can be flattered rather than satisfied.

**This project derives the process-level reward from explicitly recovered argument structure instead.** A reasoning trace is split into steps, an argument relation model identifies which steps support, contradict, or restate which others, and deterministic graph measurements over the resulting structure form the reward. No judge model, no reward-model training, no labelled reasoning data.

Stated precisely, the reward signal is:

- **general-purpose**, applying to reasoning on ordinary benchmark tasks rather than to argument writing;
- **deterministic and inspectable**, computed by graph algorithms over a formal representation rather than produced by a model;
- **independent of ground truth**, computable for any trace whether or not the answer is known.

No published work occupies that combination. That is the contribution, and it is stated once here and not inflated elsewhere.

### 1.1 Positioning

Two axes organise the nearest work. The first is **how structure enters the model**: by supervised training on structure-building as a task, or by reward during reinforcement learning. The second is **where the signal comes from**: a learned or judged score, or recovered symbolic structure.

|                            | Learned / judged signal                    | Recovered symbolic structure    |
| -------------------------- | ------------------------------------------ | ------------------------------- |
| **Supervised fine-tuning** | standard process supervision               | Ryu et al. 2026; Li et al. 2026 |
| **Reinforcement reward**   | Rubrics as Rewards; Ziegenbein et al. 2026 | **this project**                |

Ziegenbein et al. (2026) is the closest neighbour: reinforcement learning with a multi-component argument-quality reward. It differs on both axes that matter here, since their signal comes from trained classifiers and their task is argument editing rather than general reasoning.

---

## 2. Research questions and hypotheses

Hypotheses are recorded now so that the analysis cannot be reverse-engineered from the result.

**RQ1. Can argument structure be recovered from chain-of-thought traces with usable fidelity?**
Argument mining models are trained on essays and debate transcripts. A reasoning trace is a different register: shorter steps, arithmetic, enumeration, no speakers. Whether the relation model behaves sensibly on it is an empirical question, not an assumption.
_H1: the model produces connected, plausible structures for a majority of traces, with characterisable failure modes._

**RQ2. Does argumentative structure predict answer correctness before any training?**
If well-structured reasoning is associated with correct answers in an untrained model, there is a signal worth optimising. If not, the premise of the reward is in doubt and must be reported as such.
_H2: a positive but modest association; the individual measurements differ in how much they predict._

**RQ3. Does adding the structural reward to outcome-based reinforcement learning improve held-out accuracy?**
_H3: yes, a small to moderate improvement._

**RQ4. Does any improvement come from argument structure specifically, or merely from the reward being denser?**
This is the question that separates a result from an artefact, and it is why the ablation exists.
_H4: the full structural reward outperforms the reduced one, indicating the content matters and not only the density._

**RQ5. How does a model exploit a symbolic structural reward?**
Because the reward is a diagram rather than a number from a black box, its exploits are legible. This question is answerable whatever RQ3 returns.
_H5: exploitation concentrates on restatement and on inflating step count, both of which the design is intended to resist._

---

## 3. The pipeline, end to end

Each stage is one module in the repository. The boundaries are drawn so that each can be tested alone.

```
BBH question
     |
     v
policy model generates a chain-of-thought trace          train/grpo_train.py
     |
     v
trace split into numbered reasoning steps                reward/xaif_build.py
     |
     v
every step pair classified: supports / contradicts /
restates / nothing                                       reward/ari.py
     |
     v
directed graph: nodes are steps, edges are relations     reward/features.py
     |
     v
six measurements, each a fraction in [0,1]               reward/features.py
     |
     v
weighted sum, length-penalised, added to correctness     reward/scorer.py
     |
     v
reward returned to the training loop
```

### 3.1 Splitting the trace into steps

The relation model adds links between statements; it does not produce the statements. In the oAMF framework a separate segmentation module does that job, but those modules are trained on essays and debate.

We split the trace ourselves: newline-delimited first, since that is how models write chains of thought, falling back to sentence splitting for single-paragraph traces, with numbering stripped and fragments below fifteen characters discarded.

This is a methodological improvement rather than a shortcut. Any structure measured is then attributable to one model's errors instead of two models' errors compounding, which makes the RQ1 characterisation interpretable. Recorded as D-001.

### 3.2 Identifying relations

The model is `raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L`, the checkpoint behind the oAMF ARI module. It takes two statements and returns one of three labels with a confidence:

| Label     | Meaning                       | Stored as |
| --------- | ----------------------------- | --------- |
| Inference | the first supports the second | RA        |
| Conflict  | the two contradict            | CA        |
| Rephrase  | the two restate each other    | MA        |

Confidence floors are taken verbatim from the published module: 0.9 for Inference, 0.7 for Conflict and Rephrase. Pairs below the floor produce no link. **These thresholds are not tuned.** Adjusting a supervisor's published thresholds and then reporting an improvement would make the result impossible to defend.

The checkpoint is loaded directly rather than called as a web service, because the framework's local deployment requires Docker (which Colab does not provide) and a network call per pair would make training unaffordable. The pairing and thresholding logic is reimplemented faithfully, including the concatenation of the two statements into a single input string, which is what the published module does. Recorded as D-002.

**Verification requirement.** Before any result is reported, a sample of at least twenty traces is scored both by this implementation and by the published web service, and agreement is reported. This converts "reimplemented" into "reimplemented and verified equivalent", and the check is a stated deliverable of Week 1.

### 3.3 The pairing decision, and why it dominates the compute budget

The relation model scores pairs, one forward pass each. The number of pairs depends on the window setting:

| Setting         | Pairs from n steps | n = 15 | n = 30 |
| --------------- | ------------------ | ------ | ------ |
| all pairs       | n(n−1)/2           | 105    | 435    |
| neighbours only | n−1                | 14     | 29     |

All-pairs can see a step that supports the conclusion from five steps earlier, which is precisely what the connectivity and chain-depth measurements are designed to detect. Neighbours-only cannot see it at any price.

**This is measured, not assumed.** In Week 1 both settings are timed on a sample of real traces and the cost per training step is computed. All-pairs is used if affordable; if not, the cheaper setting is adopted with the loss of long-range detection stated explicitly as a limitation. The decision and its evidence are recorded in `run_log.md`.

### 3.4 The six measurements

All are fractions in [0,1]. Let _n_ be the number of steps and _c_ the conclusion.

| Name               | What it measures                                 | Why it is in the reward                             |
| ------------------ | ------------------------------------------------ | --------------------------------------------------- |
| `support_density`  | support links relative to n−1                    | how much of the trace does inferential work         |
| `attachment`       | steps participating in any support link          | detects floating assertions connected to nothing    |
| `connectivity`     | steps with a support path to _c_                 | does the reasoning actually lead to the answer      |
| `support_depth`    | longest support chain ending at _c_, capped at 5 | rewards chained inference over flat assertion lists |
| `conflict_rate`    | contradiction links relative to n−1              | detects self-contradiction within a trace           |
| `restatement_rate` | restatement links relative to n−1                | detects padding: the same point made twice          |

The last two enter the reward inverted, since more of either is worse.

**Every measurement is continuous, and that is a requirement rather than a preference.** GRPO computes each attempt's advantage relative to the mean of its group. If all attempts in a group score identically, every advantage is zero and the model learns nothing from that question. Binary measurements make ties frequent; fractions make them rare. A binary reward here would fail silently, in a way indistinguishable from "the idea did not work".

**Degenerate guard.** Traces with fewer than three steps, or with no identifiable conclusion, score zero on all measurements. Such traces have no structure to assess, and scoring them on an absent graph would inject noise.

### 3.5 From measurements to reward

```
structural = (weighted mean of the active measurements) x length_penalty
reward     = correctness + lambda x structural
```

where `correctness` is 1 or 0, `lambda` is 0.5, and

```
length_penalty = min(1, length_target / trace_length)
```

**Weights come from the correlation study (E1) and from nowhere else.** Measurements that do not predict correctness receive weight zero. `reward/scorer.py` raises an error rather than running with unset weights, because hand-setting them and then reporting an improvement is the single most damaging thing that could happen to this project's credibility.

**Every threshold is a percentile of the untrained model's own output**, not a chosen constant. `length_target` is the 75th percentile of baseline trace length. This is calibration to the observed distribution, and it is stated as such in the method chapter.

The length penalty exists because all six measurements are fractions, so producing more text tends to create more links. Without it, verbosity is a free way to raise the score.

---

## 4. Experiments

| ID  | Name                             | Depends on | Answers  | Output                                                |
| --- | -------------------------------- | ---------- | -------- | ----------------------------------------------------- |
| E0  | Baseline trace corpus            | pipeline   | all      | 800–1200 scored traces                                |
| E1  | Correlation and characterisation | E0         | RQ1, RQ2 | association analysis, fidelity report, frozen weights |
| E2  | Training comparison              | E1         | RQ3, RQ4 | 3 conditions x 2 seeds                                |
| E3  | Reward-hacking analysis          | E2         | RQ5      | exploit taxonomy                                      |

### E0. Baseline corpus

Qwen 2.5 3B Instruct, temperature 0.7, one trace per item across the selected task families of BIG-Bench Hard. Stored per trace: prompt, trace, extracted answer, correctness, extraction success, all step splits, all relations with confidences, all six measurements, and decoding parameters.

Answer extraction uses a forced final-answer format with a regular-expression fallback. **Extraction failures are counted separately and never silently scored as wrong**, since conflating "the model did not answer" with "the model answered incorrectly" corrupts every accuracy figure downstream.

### E0.1 Data splits: four tiers at increasing distance from training

Declared in `docs/splits.json`, frozen and committed before any model sees data, and fingerprinted so that the run log shows immediately if two runs were not comparable.

| Tier     | Content                              | What it measures                    |
| -------- | ------------------------------------ | ----------------------------------- |
| Train    | 80% of items from 6 families         | learning                            |
| Held-out | the remaining 20%, **same families** | new questions, familiar reasoning   |
| Transfer | 4 **entirely unseen families**       | new reasoning types, same benchmark |
| GSM8K    | a separate benchmark, 250 items      | cross-benchmark generalisation      |

Only the training tier is ever trained on. The item-level split is computed deterministically from a seed recorded in the split file, so exact question lists are reproducible without committing the data itself.

**The gap between held-out and transfer is the interesting comparison.** It separates having learned the questions from having learned to reason. The gap between transfer and GSM8K separates within-benchmark generality from actual generality.

**Training families** (6): formal fallacies, logical deduction (three and five objects), web of lies, disambiguation QA, navigate. Selected on a criterion stated before any reward was computed: a family is eligible only if its chains of thought contain natural-language inferential steps.

**Transfer families** (4): date understanding, penguins in a table, reasoning about coloured objects, temporal sequences. Deliberately mixed in reasoning type, so that transfer is not measured against a near-duplicate of the training set.

**Exclusions** (17 families), each with a stated reason rather than a silent omission:

- _Symbolic or mechanical_ (9), including word sorting, arithmetic, object counting, Dyck languages and the shuffled-object tasks. Their reasoning is symbol manipulation, so an argument graph over such a trace is degenerate and scoring it would measure noise.
- _Chain-of-thought underperforms_ (3): causal judgement, ruin names, snarks. Suzgun et al. (2023) report chain-of-thought prompting at or below direct prompting on these. Including them would confound a reasoning-quality intervention with tasks where reasoning aloud actively hurts.
- _Weak or redundant_ (5): minimal inferential chain, or near-duplicates of a training family.

Excluding roughly two thirds of the benchmark is a substantial scoping decision and is presented as one. The alternative, training an argument-structure reward on word-sorting traces, would produce a measurement of nothing.

### E0.2 GSM8K as an external evaluation set

Two hundred and fifty grade-school maths word problems, evaluation only, never trained on. It is included because within-benchmark transfer is a weaker claim than it appears: unseen BBH families still share format, prompt style, and answer conventions with the training families.

GSM8K shares none of those, and its reasoning is arithmetic rather than verbal. It is therefore a deliberately hard test of the central thesis. If argumentative structure improves reasoning generally rather than improving performance on one benchmark's conventions, the effect should survive the move. If it does not survive, that boundary is itself a finding worth reporting precisely.

Cost is negligible: inference on finished checkpoints, no additional training.

### E1. Correlation and characterisation

**Pre-registered analyses, in this order:**

1. Point-biserial correlation of each measurement against correctness, per family and pooled, with confidence intervals.
2. Logistic regression: `correct ~ measurements + trace_length + task_family`. **Trace length is included as a covariate deliberately**, because longer traces have more steps and therefore more links; without controlling for it, any association could be verbosity rather than structure. This is the first objection an examiner will raise.
3. Area under the ROC curve for the composite score as a predictor of correctness, compared against length alone. The comparison against length is the one that matters.
4. Relation fidelity: on thirty hand-annotated step pairs, agreement between the model's labels and manual annotation. Reported as a bound on what the reward can perceive, not as a claim about the model's general quality.

**Interpretation rules, fixed now:**

| Composite AUC, length controlled | Reading                   | Action                                                                                                  |
| -------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| > 0.60                           | meaningful association    | proceed as planned                                                                                      |
| 0.55 – 0.60                      | weak association          | proceed, temper H3                                                                                      |
| < 0.55                           | no predictive association | proceed, reframe E2 as testing whether optimisation _creates_ structure the base model does not exploit |

A null result here is a finding, not a failure, and the reframing is written before the number is seen precisely so that it cannot be motivated by the number.

**Output:** `reward/calibration.yaml`, frozen and committed. Weights set from the analysis, `length_target` set to the 75th percentile of baseline length. Any subsequent change is a deviation entry.

### E2. Training comparison

Three conditions, identical in data, steps, and every hyperparameter. Only the reward differs.

| Condition  | Reward                                                              | Seeds |
| ---------- | ------------------------------------------------------------------- | ----- |
| Baseline   | correctness only                                                    | 2     |
| Structural | correctness + λ x structural                                        | 2     |
| Ablation   | correctness + λ x structural, contradiction and restatement removed | 1     |

The ablation answers the objection that any denser reward would help. If the structural condition beats the baseline but ties with the ablation, the honest conclusion is that density helped and argument content is unproven. That conclusion is written here in advance so that it can be reported without reluctance.

Training: Qwen 2.5 3B, 4-bit quantised with low-rank adapters, group size 8, 400 update steps, learning rate 1e-5. Logged every 10% of steps: all six measurement means, KL divergence, mean trace length, held-out accuracy on a fixed probe set, and twenty saved traces from a fixed prompt panel for the qualitative comparison.

**The prompt panel and checkpoint schedule are fixed before the first run**, because the trace-evolution comparison cannot be reconstructed afterwards.

### E3. Reward-hacking analysis

Automatic screens across training checkpoints: trace length growth, connective-phrase frequency, steps per hundred tokens, divergence between support density and attachment, and restatement rate.

Manual pass: ten to fifteen traces from the structural condition's final checkpoint, hand-inspected and classified into a named taxonomy with examples.

The anticipated exploits, named in advance: restating steps to inflate the graph; fragmenting one claim into several short steps; padding with connective phrases that read as inferential; and producing shallow wide structures rather than chains. The design resists each of these (restatement is penalised, the length penalty caps padding, attachment falls when steps are fragmented), and E3 tests whether that resistance holds.

---

## 5. Evaluation

**Primary.** Exact-match accuracy across all four tiers: held-out items, transfer families, and GSM8K. Reported as a single table, since the _shape_ of the decline across tiers is more informative than any one number. All of these are independent of the training signal, which avoids the circularity of evaluating a reward with itself.

**Secondary.** Logical-consistency rate on a HaluEval subset, off-the-shelf and inference-only.

**Supporting.** Argument diagrams of the same prompt's trace at the first, middle, and final checkpoints, presented side by side. This figure is only possible because the reward is a structure rather than a scalar, and it is the clearest single illustration of what the method does.

**Statistics.** The unit of analysis is the seed. With two seeds per condition, statistical power is minimal and this is stated rather than obscured: **effect sizes with confidence intervals are reported first, and p-values are reported as supporting evidence, never as a verdict.** Claiming significance from two seeds would be indefensible; reporting the observed difference with an honest account of the uncertainty is not.

---

## 6. Gates

Each gate has a decision and a pre-planned response. Deciding early is project management; discovering late is a crisis.

| Point         | Gate                                                       | If it fails                                                                                                                   |
| ------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| End of Week 1 | training loop completes 50 steps with a placeholder reward | abandon the training experiment; submit E0 and E1 as a full study with a feasibility analysis. Still a complete dissertation. |
| End of Week 1 | relation model verified against the published service      | investigate before any corpus is generated                                                                                    |
| End of Week 1 | pairing cost measured                                      | adopt neighbours-only and state the limitation                                                                                |
| End of Week 2 | E1 written up, weights frozen                              | the insurance contribution is banked regardless of what follows                                                               |
| End of Week 3 | all training runs complete                                 | no new experiments after this point, without exception                                                                        |

---

## 7. Threats to validity

Stated here so that they appear in the dissertation as anticipated rather than as concessions extracted in a viva.

**Verbosity confound.** More text produces more steps, more links, and higher measurements. Addressed by the length penalty in the reward and by including length as a covariate in E1. It is mitigated, not eliminated.

**Relation model domain shift.** The model is trained on essays and debate, not on reasoning traces. This is RQ1 rather than an assumption, and its measured fidelity bounds everything the reward can perceive.

**Faithfulness.** Turpin et al. (2023) show that a model's stated reasoning does not necessarily reflect the computation that produced its answer. Better-structured reasoning therefore does not entail more honest reasoning. This project measures structure and claims structure, nothing more. This limit is acknowledged in the discussion rather than left for someone else to raise.

**Statistical power.** Two seeds. Effect sizes reported, significance not claimed.

**Reimplementation risk.** The relation logic is reimplemented rather than called as published. Addressed by the Week 1 equivalence check against the real service.

**Scale.** A 3B model on one benchmark. Generality is not claimed beyond it.

---

## 8. Reproducibility

- Every run records its commit reference, configuration, seed, and result in `run_log.md`, at the time it completes.
- Data splits, prompts, and the frozen calibration are committed before use.
- Seeds are fixed and stated.
- `scripts/reproduce.sh` regenerates the headline comparison from committed configurations.
- The reward pipeline has unit tests, including one asserting that a padded trace cannot outscore a clean one. If that test ever fails, the reward is exploitable.

## 9. Explicitly out of scope

Listed so that they are visibly decisions rather than omissions: no 7B model unless Week 3 finishes early; no comparison against a much larger reference model; no language model used as a judge; no human evaluation; no hyperparameter search; no custom implementation of the training algorithm; no learned reward components.
