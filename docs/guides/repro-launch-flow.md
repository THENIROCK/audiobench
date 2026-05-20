# Reproducible controversy launch flow

Use this when you want to publish one controversy-grade claim from `ab/asr-hallucination` and keep it auditable.

## What counts as publishable

`audiobench` now runs a detector pipeline on every `ab/asr-hallucination` run:

- bootstrap confidence intervals over per-domain hallucination uplift
- Benjamini-Hochberg multiple-testing correction across detector outputs
- deterministic discovery/holdout split for replication checks

Each finding gets a status:

- `validated`: passes discovery gate and holdout replication
- `candidate`: passes discovery gate, holdout not yet confirmed
- `rejected`: fails significance, support, or reproducibility checks

Only runs with at least one `validated` finding are marked `publishable`.

## 1) Run the suite and save JSON artifacts

```bash
audiobench run ab/asr-hallucination --model whisper-tiny --output results/hallucination-whisper-tiny.json
```

Repeat for any model you want to compare:

```bash
audiobench run ab/asr-hallucination --model my-asr-model --output results/hallucination-my-model.json
```

The CLI summary prints:

- ranked findings (effect size, CI, adjusted p-value)
- validation gate counts (`validated`, `candidate`, `rejected`)
- publishable/not-publishable badge

## 2) Compare competing runs

```bash
audiobench compare \
  results/hallucination-whisper-tiny.json \
  results/hallucination-my-model.json
```

`compare` shows both headline hallucination deltas and each run's top finding status, so you can avoid presenting a weaker candidate as if it were validated.

## 3) Verify reproducibility checklist before publishing

In the run JSON, confirm:

- `validation_summary.reproducibility_checklist` is all `true`
- `top_finding.status` is `validated` (not just `candidate`)
- `findings_methods` records bootstrap sample count, confidence, and correction method

If any checklist field is false, do not publish the claim.

## 4) Re-run to confirm determinism

Re-run the same command and compare hashes:

```bash
audiobench run ab/asr-hallucination --model whisper-tiny --output results/hallucination-whisper-tiny-rerun.json
```

For the same model/config/seed and identical per-clip outputs, `run_hash` should match. If not, treat the finding as not launch-ready.

## 5) Push validated artifacts

```bash
hf auth login
audiobench push results/hallucination-my-model.json --pretty-json
```

Leaderboard records carry the top finding status and validated-finding count, so downstream readers can filter to reproducible claims instead of raw score alone.
