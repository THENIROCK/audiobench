# Reproducibility guarantees

audiobench is designed so that two people running the same command on different machines get the same `run_hash`.

## What's pinned

- Manifest, mixture, and probe seeds are fixed at the suite level.
- The mixer is deterministic (peak-normalize, RMS-match, sum).
- The prompt set is versioned and pinned in `run_hash` (`prompt_version`, `parser_version`, `prompt_ensemble`, plus a SHA-256 over the canonicalized paraphrase list).

## Run JSON artifact

Every `audiobench run` writes a JSON artifact with:

- `suite`, `revision`, `model`, `seed`, `config`
- per-clip / per-mixture hypotheses (with per-paraphrase answers when ensembling)
- per-condition metrics and a weighted mean
- `run_hash` — SHA-256 over the canonicalized run payload, including mixture spec and prompt config

The `run_hash` is the single value that uniquely identifies a run. If two runs have the same hash, the input audio, prompts, model adapter version, and seeds were identical and you can compare numbers fairly. If they don't match, `audiobench compare` will tell you exactly which field disagrees.

## Sharing a run

`audiobench push` is suite-agnostic and works for both suites; it only reads `suite`, `revision`, `run_hash`. In the MVP it's a local-only "push" stub that prints a signed payload (suite, revision, run_hash, payload_sha256). No network traffic.

```bash
audiobench push results/sound-id-heuristic.json --pretty-json
```

## What's NOT pinned

- **Model weights for non-bundled adapters.** If `clap-base` or `qwen2-audio-7b` upstream re-uploads checkpoints under the same name, your numbers can drift. Pin the checkpoint hash in your own pipeline if this matters to you.
- **Hardware non-determinism.** Most Qwen2-Audio inference paths run with `do_sample=False` (deterministic), but some accelerators (especially with reduced-precision attention) can produce slightly different logits between runs. The yes/no parser is permissive enough that this rarely flips a decision; if it does, you'll see it in the per-probe JSON.
- **Resampling and codec edge cases** when you supply your own data through a pack. The same `.wav` file fed through different host-side resamplers can produce different fingerprints.

## Compare semantics

`audiobench compare a.json b.json` per-suite:

- `ab/asr-robust`: lower WER wins, computed per condition and on the weighted mean.
- `ab/sound-id`: higher recall wins, lower FPR wins, F1 is reported alongside as a single-number summary.

By default, `compare` refuses to compare two `ab/sound-id` runs whose prompt configs disagree. Use `--allow-mismatched-prompt` to override; the comparison header will annotate the mismatch.
