# Suites

A **suite** is a benchmark: a manifest of clips or stimuli, a set of conditions
or perturbations, and a per-suite scoring rule. Every suite is identified by a
stable `suite_id` (e.g. `ab/sound-id`) and ships with a fixed `revision`. Both
are recorded in every run JSON.

Suites are grouped into three families that test different things and ship
their own adapter contracts.

## Task suites

Behavioral evaluations on labeled clips. The model is a transcriber,
identifier, or detector.

| Suite | Task | Default model | Headline |
|---|---|---|---|
| [`ab/asr-robust`](asr-robust.md) | Speech recognition under noise / bandlimit / reverb | `whisper-tiny` | `weighted_mean_wer` |
| [`ab/asr-hallucination`](asr-hallucination.md) | Non-speech ASR hallucinations with validated findings | `whisper-tiny` | `weighted_hallucination_rate` |
| [`ab/sound-id`](sound-id.md) | Sound-event identification on labeled mixtures | `heuristic-v0` | `components_understood / components_present` |

## Signal suites

Reference-aware fidelity / psychoacoustic / phase checks. The model is an
[`AudioProcessor`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/audio_processor.py)
that takes `(audio, sample_rate)` and returns processed audio.

| Suite | What it stresses | Headline |
|---|---|---|
| [`ab/fidelity-roundtrip`](fidelity-roundtrip.md) | SI-SDR, MR-STFT, true peak, LUFS drift across stimuli × conditions | `weighted_si_sdr_db`, `max_true_peak_dbtp` |
| [`ab/psychoacoustic-masking`](psychoacoustic-masking.md) | Audibility — keep audible tones audible, leave masked tones masked | `masking_respect_score` |
| [`ab/phase-coherence`](phase-coherence.md) | Polarity, inter-channel correlation, M/S round-trip, sub-sample delay | `phase_coherence_score`, `mean_polarity_score` |

## Temporal task suites

Frame-level evaluations. The model emits time-stamped events or speaker turns.

| Suite | Task | Headline |
|---|---|---|
| [`ab/sed-urban`](sed-urban.md) | Sound event detection on labeled urban-noise soundscapes | `event_f1_iou50`, `segment_f1_1s` |
| [`ab/diarization-cw`](diarization-cw.md) | Speaker diarization with DER + Hungarian alignment + 0.25 s collar | `der`, `mean_speaker_count_error` |

## Discover what's available

```bash
audiobench list                              # every suite + status
audiobench info ab/fidelity-roundtrip        # stimuli, conditions, adapters
audiobench list-models --suite ab/sed-urban  # bundled SED adapters
```

## Reproducibility model (shared across all suites)

- Every fixture is procedurally rendered or hashed from a known source. The
  manifest digest is part of every `run_hash`.
- The CLI seed is folded into the run hash. Two runs with the same suite +
  revision + manifest + model + seed produce the same hash.
- `audiobench compare` and `audiobench gate` work uniformly across suites —
  same artifact schema, same exit-code semantics, same JUnit output.

See [Reproducibility guarantees](../reference/reproducibility.md) for the
hash schema and replay procedure.

## Adding a new suite

A suite is a Python module under `src/audiobench/suites/` that exposes:

- a `SUITE_ID` and `SUITE_REVISION` string,
- a `load_manifest() -> dict` for `audiobench info`,
- a `run_suite(*, model_name, seed, ...) -> dict` runner that returns the
  canonical artifact shape (`suite`, `model`, `headline`, `per_*`,
  `run_hash`, `manifest_hash`).

Register it in `audiobench/suites/__init__.py`, add adapter-contract / report
/ gate / matrix / compare wiring for the new headline, and write tests
following `tests/test_signal_suites.py` or `tests/test_temporal_suites.py` as
templates.
