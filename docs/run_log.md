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

*No runs yet. The repo scaffold was built 2026-07-20; the first entry will be
the oAMF spike (see docs/oamf_spike.md).*
