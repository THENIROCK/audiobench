# Suites

A **suite** is a benchmark task: a manifest of clips or mixtures, a set of conditions or perturbations, and a per-suite metric. Every suite is identified by a stable `suite_id` (e.g. `ab/sound-id`) and ships with a fixed `revision`. Both are recorded in every run JSON.

| Suite | Task | Default model | Conditions |
|---|---|---|---|
| [`ab/asr-robust`](asr-robust.md) | Speech recognition under perturbations | `whisper-tiny` | `clean`, `noise-cafe-10db`, `noise-pink-5db`, `bandlimited-8k`, `reverb-medium` |
| [`ab/sound-id`](sound-id.md) | Sound-event identification on labeled mixtures | `heuristic-v0` | `solo`, `pair`, `triple`, `quad` (mixture sizes) |

List the suites the current build knows about:

```bash
audiobench list
audiobench info ab/sound-id
audiobench info ab/asr-robust
```

## Why two suites

The two suites stress different things:

- `ab/asr-robust` is about **graceful degradation under perturbation**. Same content, harder channel. WER under noise tells you whether a model has actually learned a robust acoustic representation or has just overfit to clean studio audio.
- `ab/sound-id` is about **multi-label identification under polyphony**. As mixture size grows from `solo` to `quad`, the model has to disentangle more concurrent sources. The headline metric (`components understood: X / Y`) makes the polyphony cost legible at a glance.

Both suites share the same run-artifact schema, the same `compare` command, and the same `run_hash` semantics. Adding a third suite (say `ab/diarization`) is mostly authoring a new module and registering it in the runner.
