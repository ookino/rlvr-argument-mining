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
