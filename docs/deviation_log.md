# Deviation log

Every departure from `experiment_plan.md`, recorded on the day it happens.
Backfilling this defeats its purpose. Examiners read a maintained deviation
log as maturity, and a silent omission as a gap.

### D-001  2026-07-20  We split traces into steps ourselves
- **Planned:** use an oAMF segmenter module to split traces into statements.
- **Actual:** we split the trace ourselves (`reward/xaif_build.py`) and hand
  the relation model a ready-made list of steps.
- **Reason:** the relation model only adds links between statements; it does
  not split text. Segmenters are trained on essays and debates, whereas a
  chain of thought is already written in steps and splits reliably on
  newlines. Doing it ourselves removes a noisy stage.
- **Effect on claims:** improves them. Any structure measured is now
  attributable to one model's errors rather than two compounding, which makes
  the characterisation study cleaner.

### D-002  2026-07-20  Relation model loaded directly, not via the oAMF service
- **Planned:** call the oAMF ARI module through the framework's routing.
- **Actual:** load `raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L` directly with
  transformers, reimplementing the module's pairing and thresholding logic
  (published thresholds kept verbatim).
- **Reason:** oAMF's local routing deploys modules via Docker
  (`oamf/deployer/deploy.py` shells out to `docker-compose build` and `up -d`).
  Colab provides no Docker daemon, so local routing is impossible there, and
  web-service calls inside a training loop are too slow and add a network
  dependency. The model is a public checkpoint, so this is the same weights
  and the same behaviour without the container.
- **Effect on claims:** none on validity. oAMF and the ARI module are still
  cited as the source of the model and the method.

### D-003  2026-07-20  Scope reduced to fit a four-week build
- **Planned:** 7B model, scale comparison against a ~70B reference,
  LLM-as-judge secondary metric, full BBH coverage, three seeds.
- **Actual:** 3B model (7B a stretch goal), no scale comparison, no judge
  metric, reduced family coverage, two seeds on the headline comparison.
- **Reason:** available time reduced to four weeks. Cuts chosen to protect the
  comparison that carries the thesis and the correlation study that stands
  without it.
- **Effect on claims:** narrows generality. Stated openly in Limitations.

### D-004  2026-07-20  GSM8K added as an external evaluation set
- **Planned:** train and evaluate entirely within BIG-Bench Hard.
- **Actual:** training unchanged; GSM8K (250 items, test split) added as a
  fourth evaluation tier, inference only.
- **Reason:** raised by the supervisor. Within-benchmark transfer is a weaker
  claim than it looks, since unseen BBH families still share format, prompt
  style and answer conventions with the trained ones. GSM8K shares none of
  these and its reasoning is arithmetic rather than verbal, so it is a genuine
  test of whether the effect is about reasoning or about one benchmark's
  conventions.
- **Effect on claims:** strengthens them. The generalisation claim becomes
  cross-benchmark rather than within-benchmark. Cost is negligible: no extra
  training runs.

### D-005  2026-07-20  Held-out tier redefined
- **Planned:** "held-out families" and "transfer families" as separate tiers.
- **Actual:** held-out is now 20% of items from the SAME families used for
  training; transfer remains entirely unseen families.
- **Reason:** as originally written the two tiers both measured unseen
  families, so they tested the same thing twice. The revision gives four tiers
  at increasing distance from training, and makes the held-out to transfer gap
  interpretable as the difference between learning the questions and learning
  to reason.
- **Effect on claims:** none invalidated. No data had been generated.

### D-006  2026-07-20  Training widened to 8 files; LogiQA added as a fifth tier
- **Planned:** 6 BBH training files (~1200 questions), four evaluation tiers.
- **Actual:** 8 training files (~1600 questions) after adding logical deduction
  at seven objects and sports understanding; LogiQA added as an
  evaluation-only tier alongside GSM8K.
- **Reason:** the training set was small in absolute terms, and breadth was
  raised by the supervisor. Breadth is added to EVALUATION rather than to
  training, deliberately: anything added to training is lost as a test, and
  the question at issue is generalisation. GSM8K and LogiQA pull in different
  directions, arithmetic and argumentative respectively, so the pair bounds
  where the effect holds.
- **Effect on claims:** strengthens the generalisation claim. Training remains
  a single distribution, which keeps the experimental design clean.
- **Note:** the 8 training files are 6 distinct task types, since logical
  deduction appears at three sizes. Stated openly rather than counted as 8.

### D-007  2026-07-27  Training uses plain TRL, not Unsloth
- **Planned:** train with Unsloth (4-bit QLoRA plus fast GRPO), per the code
  index and requirements.
- **Actual:** train with plain TRL (transformers + peft + bitsandbytes for the
  4-bit QLoRA, TRL's GRPOTrainer for the loop). Unsloth is not imported.
- **Reason:** the installed Unsloth (2026.7.5, with trl 0.24 and transformers
  5.5) crashes inside its own compiled GRPO trainer: an off-by-one in the
  log-prob step (index [257,1] vs [256,vocab]). It fails under torch.compile
  and also in eager mode, so it is a real bug in that version, not a config
  error. Guessing at an older compatible Unsloth version means repeated
  reinstall-and-restart cycles with no guarantee.
- **Effect on claims:** none on the science. The training algorithm (GRPO),
  model (Qwen 2.5 3B), and 4-bit QLoRA setup are unchanged; only the library
  wrapper differs. The one practical cost is generation speed, which Unsloth
  would have accelerated. For real training that speed is recovered with vLLM
  (use_vllm in TRL's GRPOConfig), which is independent of Unsloth. To be
  revisited when moving from the smoke test to real runs.

### D-008  2026-07-29  Conclusion node = last reasoning step, not the answer line
- **Planned:** the conclusion is the trace step matching the extracted answer.
- **Actual:** trailing answer-announcement lines ("The answer is X") are dropped
  from the steps, and the conclusion is the last remaining reasoning step. The
  answer is still extracted separately from the raw text, so correctness is
  unaffected.
- **Reason:** on the E0 corpus, 93% of traces had the conclusion detected as the
  bare answer line, and 39% had connectivity = 0 because the relation model
  never links the reasoning to "The answer is No". So the two conclusion-based
  measures (connectivity, support_depth) were measuring a disconnected node.
- **Effect on claims:** corrects two of the six measures. The E2 correlation
  study is re-run on the re-scored corpus before any conclusion is drawn about
  whether structure predicts correctness. The corpus now also stores the raw
  relations, so future feature changes can be tested without re-running ARI.

### D-009  2026-07-30  Base model = a reasoning model (Qwen3), not an instruct model
- **Planned:** generate traces with Qwen 2.5 3B Instruct and prompt it to reason
  step by step.
- **Actual:** switch to a native reasoning model - Qwen3-4B for the corpus,
  Qwen3-1.7B for the lighter training runs - which produces its chain of thought
  inside <think>...</think> tags. Trace extraction now mines the reasoning from
  inside the tags and reads the answer from after them (reward/xaif_build.py,
  _split_think). Generation budget raised (max_new_tokens 512 -> 2048) because
  reasoning models think for much longer before answering.
- **Reason:** supervisor was firm that the project (title: "Reasoning LLMs")
  must use an actual reasoning model. An instruct model's chain of thought is
  prompt-induced and shallow, so structure measured on it is not representative
  of how a reasoning model reasons.
- **Effect on claims:** the pipeline downstream of extraction (ARI, the six
  features, the scorer, the correlation and fidelity analyses) is unchanged and
  re-runs on the new corpus. The Qwen2.5-3B-Instruct corpus, its E2 null, and
  the RQ1 fidelity pilot are retained as an INSTRUCT-MODEL BASELINE; the Qwen3
  corpus becomes the reasoning-model comparison. Extraction is backward
  compatible (no <think> tags -> old behaviour), so the baseline still scores.

### D-010  2026-07-30  Propositionalisation: test both arms, don't assume
- **Question raised with supervisor:** should a propositionalisation stage run
  before ARI, to turn raw steps into clean proposition (I-)nodes in xAIF, the
  format the argument-mining pipeline is built around? (Related to D-001, where
  we already do our own segmentation rather than use oAMF's.)
- **Decision:** do not assume it either way. Run BOTH and decide by measured
  fidelity:
  - Arm A (direct): trace -> our steps -> ARI's prediction pipeline (current).
  - Arm B (propositionalise): trace -> clean proposition nodes -> ARI.
  Score each arm with the RQ1 fidelity harness (annotate_pilot.py). Keep
  whichever gives higher precision (fewer spurious links); report the gap.
- **Rationale:** the supervisor's read was that self-generated reasoning traces
  may already be clean enough to feed ARI directly, so the extra stage might not
  be needed - but it is an empirical question, not an assumption. This also
  directly tests the over-generation finding: if Arm B removes the spurious
  header/framing links, precision should rise.
- **Effect on claims:** turns a skipped-step assumption into a measured design
  choice. Sequencing: run Arm A on the first Qwen3 corpus, look at precision,
  then build Arm B only if Arm A is not already clean.

### D-011  2026-08-02  Conclusion node: drop trailing self-doubt, not just answer lines
- **Planned (D-008):** the conclusion is the last reasoning step after dropping
  trailing "The answer is X" lines.
- **Observed on the Qwen3 pilot:** reasoning models state their verdict and then
  keep second-guessing - "But wait, the conclusion is...", "Alternatively,
  suppose...". Those trailing lines became the conclusion node instead of the
  verdict. On a hand-read formal_fallacies trace, every real support link
  pointed into the verdict (step 24), but the conclusion was marked as a trailing
  "But wait" fragment (step 25) - so connectivity measured paths to a node
  nothing supported. This is a systematic cause of the near-zero connectivity
  (0.03) on the pilot corpus, not just trace bloat.
- **Actual:** extend the trailing-line drop to also remove trailing self-doubt
  ("But wait", "Wait", "Hold on", "Alternatively", "Let me re-check/verify/
  reconsider/..."), so the conclusion lands on the verdict. Only TRAILING lines
  are dropped; mid-trace reconsidering is real reasoning and stays. Bare "Hmm"/
  "Actually" openers are excluded as they usually carry real content.
  (reward/xaif_build.py, _REFLECTION_LINE / _is_trailing_noise.)
- **Effect on claims:** fixes connectivity and support_depth on reasoning traces.
  Verified 0/300 conclusion indices change on the instruct corpus, so the
  instruct baseline is untouched and the two corpora stay comparable. The Qwen3
  corpus must be re-scored under this rule before its connectivity is trusted.
