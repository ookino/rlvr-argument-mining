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

- **Result:** FINAL CLEAN NUMBERS PENDING the post-fix regeneration. The last
  pre-fix run gave 38% correct overall with a healthy per-family spread
  (6-58%); re-labelling with the synonym fix lifted the two low families
  (sports 17->39%, web_of_lies 6->30%) and the prompt fix is expected to lift
  them further once regenerated. Update this entry with the final table.

- **Structure feature means (pre-fix run, for reference):** support_density
  0.66, attachment 0.58, connectivity 0.30, support_depth 0.38, conflict 0.02,
  restatement 0.19. Non-degenerate and sensible; the pipeline produces real
  structure on chain-of-thought traces.

- **Next:** E2 correlation study once the clean corpus is confirmed.

---

*Biggest project risk (does training run) is now cleared. E0 corpus in
progress; correlation study (E2) is next.*
