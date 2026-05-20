# `ab/phase-coherence`

Stereo phase / multichannel coherence regressions: polarity flips, inter-channel
delay errors, broken Mid/Side round-trips. The failures here are the ones that
show up only when listeners go to mono (and silence appears) or when the
content gets summed for broadcast loudness.

```bash
audiobench run ab/phase-coherence --model passthrough
```

## What it measures

Five procedural stereo fixtures, three different check types:

| Check | Metric | Pass condition |
|---|---|---|
| `correlation` | Pearson correlation between L and R | within ±0.05 of the expected value |
| `polarity` | Per-channel polarity preservation score | ≥ 0.99 |
| `ms_roundtrip` | SNR of `(M ± S)` round-trip vs. original | ≥ 40 dB |

Metric implementations live in
[`src/audiobench/signal_metrics.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/signal_metrics.py)
(`interchannel_correlation`, `polarity_preservation_score`,
`mid_side_round_trip_snr_db`).

## Stimuli

| Stimulus | Setup | What a buggy processor would do |
|---|---|---|
| `identity-stereo` | Mono content duplicated to L and R (expected corr = +1) | Inter-channel drift caused by independent per-channel processing. |
| `polarity-pair` | L = tone, R = -tone (expected corr = -1) | Re-aligning channels (collapsing to mono) silences this stimulus. |
| `quad-phase-pair` | L and R 90° apart (expected corr ≈ 0) | Any phase coupling pulls correlation away from zero. |
| `ms-roundtrip` | Stereo built as L = M+S, R = M-S | A non-linear or non-additive processor breaks the M/S identity. |
| `sub-sample-delay` | R delayed by 0.4 samples relative to L (expected corr ≈ 0.99) | Sample-aligned re-quantization destroys the inter-aural time difference. |

Each stimulus is rendered deterministically; the manifest hash captures them
so cross-run comparisons are exact.

## How "passed" is computed

A stimulus is considered passed iff its dispatch check meets its threshold:

- **correlation** stimuli pass when
  `abs(measured - expected) ≤ CORRELATION_TOLERANCE (0.05)`.
- **polarity** stimuli pass when both per-channel polarity scores are
  `≥ POLARITY_PASS_MIN (0.99)`.
- **ms_roundtrip** stimuli pass when
  `mid_side_round_trip_snr_db ≥ MS_ROUNDTRIP_MIN_SNR_DB (40.0)`.

The thresholds are CLI-invisible constants in
[`phase_coherence.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/suites/phase_coherence.py).
Bump them locally if you need stricter or laxer policy than the defaults.

## Headline and gate keys

```json
{
  "phase_coherence_score": 1.0,
  "mean_polarity_score": 1.0,
  "passed_count": 5,
  "stimulus_count": 5
}
```

- `phase_coherence_score = passed_count / stimulus_count` — the
  one-number-fits-the-tweet headline.
- `mean_polarity_score` — average of per-stimulus polarity-preservation
  scores; useful for distinguishing "broke one stimulus" from "broke all of
  them".

Gate file keys (`gate.yaml → phase_coherence:`):

- `min_phase_coherence_score` — floor on the pass-rate (0–1).
- `min_mean_polarity_score` — floor on polarity preservation (0–1).

CLI shortcuts: `--min-phase-coherence`, `--min-polarity`.

## Adapter contract

Same [`AudioProcessor`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/audio_processor.py).
The bundled `polarity-flip-right` adapter is the canonical regression target:
it inverts the right channel, which collapses identity-stereo correlation to
-1, flips the polarity check, and breaks M/S identity.

```bash
audiobench run ab/phase-coherence --model polarity-flip-right
```

## Scope and caveats

Stereo only. Multichannel layouts (5.1, ambisonics) would need a generalized
spatial metric; that's out of scope for the MVP. Stimuli are short tones —
the suite catches systematic phase errors, not transient or modulation-rate
issues you'd find in music programme.
