# Run log

One entry per run, newest at the top. Per Workflow section 6, the entry is
written **with a prediction** before the run starts, and the outcome is logged
against that prediction afterwards. A run with no entry did not happen.

Run IDs (`E4-B-s42`) are stamped on four things and must match across all of
them: the saved config copy, the wandb run name, the output directory, and the
entry here.

## Template

```
### <run-id>  <date>
- **Config:** configs/<file>.yaml @ <git commit>
- **Host:** <local | colab | hpc>
- **Prediction:** what I expect and why, written BEFORE starting
- **Outcome:** what happened
- **Prediction vs outcome:** where I was wrong, and what that implies
- **Artifacts:** results/<run-id>/, wandb link
- **Deviations:** none, or a pointer to logs/deviations.md
```

---

### setup-checks  2026-07-27
- **What:** first run of the relation model (ARI) on a Colab L4 GPU, notebook
  `notebooks/01_setup_checks.ipynb`. Purpose was to close the three unknowns in
  the scoring pipeline before building on it.
- **Host:** Colab (L4 GPU), repo public for frictionless clone.
- **Model:** raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L, loaded direct from
  Hugging Face, `running on: cuda`.

- **Q1, relation classes:** FOUR, not three.
  `{0: No-Relation, 1: Inference, 2: Conflict, 3: Rephrase}`. The model can
  itself judge a pair unrelated, on top of our confidence thresholds. Code now
  counts "model said no relation" separately from "model guessed a relation but
  below its confidence floor" (n_model_no_relation vs n_below_threshold), a
  useful distinction for the parser-characterisation study (RQ1).

- **Q2, arrow direction:** the model returns support edges pointing
  conclusion -> premise (raven example gave 2->0, 2->1). This is the essay
  convention (claim first, support after). Chain of thought is the reverse
  shape, so we orient our support edges premise -> conclusion (earlier ->
  later). Documented as a concrete instance of the essay-to-CoT domain shift
  (RQ1). Fixed in reward/ari.py; verified the raven premises now point into
  the conclusion.

- **Q2b, open question parked:** on the raven trace the conclusion heuristic
  picked the bare answer line ("The answer is black") rather than the true
  argumentative conclusion ("therefore every bird is black"), leaving the
  chosen conclusion node disconnected. Not fixed on a synthetic example; to be
  resolved from real BBH traces at the corpus stage (E0/E1).

- **Q3, throughput:** on the L4, ~600 pairs/sec all-pairs, ~460 pairs/sec
  neighbours-only (overhead-dominated on small inputs, so a conservative
  floor). Projected ~40 min of scoring per training run at all-pairs, ~15-20%
  overhead on a 3-5 h run: affordable. Decision: keep all-pairs (window: null)
  to retain long-range link detection. Neighbours-only remains the pre-planned
  fallback. Real figure to be recorded when training runs.

---

### smoke-test (step 2)  2026-07-28
- **What:** prove the GRPO training loop runs end to end before wiring in the
  real reward. Fake (random) reward, 5 steps, toy questions, Qwen 2.5 3B in
  4-bit on a Colab L4.
- **Result:** PASS. Model loaded, adapters attached, 5 GRPO steps completed,
  "training loop finished without crashing". Loss ~0 as expected (random
  reward, KL off, so no learning signal; the goal was only that it runs).
- **Getting here took several environment fixes, all logged as deviations or
  notes:**
  - Unsloth 2026.7.5 GRPO trainer crashes (torch.compile error, then an
    off-by-one in eager). Switched to plain TRL (D-007).
  - TRL eagerly imports a broken mergekit; force-disabled it before importing
    the trainer.
  - Running a notebook dirties its own .ipynb, which blocked git pull, so
    fixes were not reaching the runtime. Setup cell now does
    fetch + reset --hard.
  - bf16 requires a GPU; added an explicit GPU check and bf16/fp16 auto-pick.
- **Speed note:** generation runs in the plain (no-vLLM) path and is slow. Real
  training will add vLLM for generation speed. Not a blocker for the smoke test.
- **Next:** step 3, generate the E0 baseline corpus with real BBH questions and
  score it with the real reward pipeline.

---

### E0 baseline corpus  2026-07-28 to 2026-07-29
- **What:** generate a chain-of-thought trace per training question with the
  untrained base model (Qwen 2.5 3B Instruct), check correctness, and score
  each trace with the argument pipeline. Input to the correlation study (E2).
- **Host:** Colab (T4/L4 GPU). Corpus saved to Google Drive so it survives a
  new runtime; generation is resumable (skips items already done).
- **Scale:** 300 traces, sampled across all 8 training families.

- **Data-quality issues found by manually inspecting real traces, each fixed
  before the corpus was trusted:**
  1. Correctness used substring matching, so "valid" matched inside "invalid"
     and opposite answers counted as correct. Inflated correctness to a fake
     86%. Fixed to whole-word matching (true value ~48% on that family).
  2. Sampling took the first N questions, all from one family. Fixed by
     shuffling with a fixed seed before sampling.
  3. A regeneration appended to the old file instead of replacing it, mixing
     old and new data. Fixed with an explicit wipe and a generate(fresh=True)
     option.
  4. The base model answers in each task's own vocabulary ("plausible" not
     "yes", "T" not "Yes"), which were marked wrong. web_of_lies came out at
     6%, below the ~50% of random guessing on a binary task, which flagged a
     labelling problem rather than difficulty. Fixed by stating the expected
     answer format per family in the prompt and accepting synonyms.

- **Method note for the write-up:** correctness labels were validated by manual
  inspection of sampled traces at each stage; four labelling/sampling issues
  were identified and corrected before use.

- **Result (clean corpus, 2026-07-29, post-fix):** 300 traces, **167/300
  correct (56%)**, 24 extraction failures. Balanced spread across families with
  no below-chance outliers, confirming the labelling fixes worked:

  | family | correct |
  |---|---|
  | disambiguation_qa | 19/43 (44%) |
  | formal_fallacies | 20/41 (48%) |
  | logical_deduction_five_objects | 10/23 (43%) |
  | logical_deduction_seven_objects | 14/37 (37%) |
  | logical_deduction_three_objects | 29/44 (65%) |
  | navigate | 32/41 (78%) |
  | sports_understanding | 26/41 (63%) |
  | web_of_lies | 17/30 (56%) |

  The two previously-broken families recovered: web_of_lies 6% -> 56%,
  sports_understanding 17% -> 63%, after the answer-format prompt + synonym fix.
  web_of_lies at 56% (a binary task) is at chance, as expected for this model.

- **Structure feature means:** support_density 0.66, attachment 0.60,
  connectivity 0.29, support_depth 0.34, conflict 0.06, restatement 0.20.
  Non-degenerate and sensible. The relatively low connectivity is useful
  variance for E2 (traces differ in how well they reach their conclusion) and
  partly reflects the parked conclusion-detection question from setup-checks.

- **Next:** E2 correlation study on this corpus.

### E2 robustness check - lower ARI thresholds  2026-07-29
- **Question:** were the strict confidence floors (support 0.9) hiding a signal
  by dropping too many links?
- **Method:** re-mined the corpus locally with lower floors (Inference 0.7,
  Conflict/Rephrase 0.5) and re-ran the correlation.
- **Result:** links per trace 16.9 -> 58.7 (3.5x more), mean connectivity
  0.35 -> 0.83. But the structure composite AUC moved only 0.48 -> 0.51, still
  at chance; length still predicts better (0.55).
- **Conclusion:** the null is NOT a thresholding artefact. Even with 3.5x more
  structure and far higher connectivity, correctness does not track structure.
  This strengthens the null.

---

### E2 correlation study (C1)  2026-07-29
- **Question:** does argument structure predict answer correctness in the
  untrained model? Analyses pre-registered in experiment_plan.md section E2.
- **Data:** the E0 corpus, re-scored after the D-008 conclusion fix
  (results/E0/corpus_rescored.jsonl). 276 traces used, 24 extraction failures
  dropped. Full output in results/E0/e2_correlation.txt.
- **Headline result: NO pooled signal.**
  - Structure composite AUC = 0.48 (below 0.5, no better than chance).
  - Length-only AUC = 0.55: trace length predicts correctness better than
    structure does, and shorter traces are more likely correct.
  - No single feature clears |point-biserial r| > 0.1 in the expected
    direction pooled. Logistic regression with length and family controlled
    finds no structure feature significant (all p > 0.4); length is the only
    marginally significant predictor (p = 0.04).
  - This is the null outcome the plan anticipated. It is ROBUST: it survived
    fixing the conclusion-detection bug (D-008), so it is a real absence of
    pooled signal, not an artefact of noisy measures.
- **The finding is the per-family heterogeneity, and the D-008 fix sharpened
  it.** In chain-structured tasks all four support measures correlate
  positively with correctness as hypothesised:
  - web_of_lies: density +0.34, attachment +0.28, connectivity +0.24, depth +0.32
  - navigate: density +0.28, attachment +0.31, connectivity +0.19, depth +0.20
  - sports_understanding: weak positive across the board
  In other tasks the effect is flat or reversed (logical_deduction_5 negative,
  disambiguation flat). Pooling averages these opposite effects to zero.
- **Interpretation (for the discussion chapter):** structure predicts
  correctness where the task itself is a reasoning chain (navigate, web of
  lies); it does not where the model produces structured-looking but
  unsound reasoning (logical deduction). A task-dependent, honest finding.
- **Consequence for the reward:** pooled, no feature earns a data-driven
  weight. Per the pre-registered plan, C3 is reframed: test whether REWARDING
  structure creates useful structure and helps, rather than assuming the base
  model already exploits it. Weights decision (uniform vs per-family-informed)
  to be taken with the supervisor.
- **Next:** supervisor discussion with the per-family table, then set weights
  and proceed to E4 training.

---

*Biggest project risk (does training run) is now cleared. E0 corpus in
progress; correlation study (E2) is next.*
