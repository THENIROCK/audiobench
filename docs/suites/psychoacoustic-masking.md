# `ab/psychoacoustic-masking`

Does the processor under test respect *audibility*? Audio that should be heard
stays heard; audio that should be inaudible doesn't get amplified into
existence.

```bash
audiobench run ab/psychoacoustic-masking --model passthrough
```

## What it measures

Every stimulus is a deliberately calibrated tone-in-noise pair where we know,
a priori, whether the target tone is psychoacoustically audible or masked.
After the processor runs, two checks fire:

- **Audible stimulus** → the in-band SNR around the target tone must stay
  above `AUDIBLE_SNR_MIN_DB` (-3 dB). The tone must not be ducked, gated, or
  spectrally smeared out.
- **Inaudible stimulus** → the in-band energy around the target frequency
  must stay within `INAUDIBLE_BAND_DELTA_MAX_DB` (3 dB) of the input's energy.
  The processor must not invent a tone (e.g. via excessive enhancement or
  hallucinated bandwidth extension).

Both checks live in the suite runner; thresholds are exposed as constants in
[`src/audiobench/suites/psychoacoustic_masking.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/suites/psychoacoustic_masking.py).
The underlying band-limited SNR / energy math lives in
[`signal_metrics.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/signal_metrics.py)
(`band_snr_db`, `band_energy_db`).

## Stimuli

Five procedurally rendered clips, deterministic, no downloads:

| Stimulus | Expected | What it stresses |
|---|---|---|
| `tone-1k-audible` | audible | 1 kHz tone +6 dB above pink noise — must remain audible after processing. |
| `tone-1k-masked` | masked | 1 kHz tone deeply buried in pink noise (~-40 dB below). Should *not* be brought out. |
| `tone-500-cross-band-audible` | audible | 500 Hz tone with a narrow-band masker at 2 kHz. Cross-band masking is weak — tone must survive. |
| `quiet-tone-4k` | audible | -40 dBFS isolated 4 kHz tone. Catches denoisers that gate below a quietness threshold. |
| `no-tone-pink-only` | masked | Pink noise alone. The processor must not hallucinate a 1 kHz tone where none exists. |

## What "respected" means in the report

Per stimulus the report shows:

```text
 stimulus                     target Hz  expected   in-band SNR dB  respected
 tone-1k-audible                  1000   audible              7.8   yes
 tone-1k-masked                   1000   masked              -33.2   yes
 tone-500-cross-band-audible       500   audible              5.6   yes
 quiet-tone-4k                    4000   audible              2.1   yes
 no-tone-pink-only                1000   masked               0.4   yes
```

- `in-band SNR dB` is computed for audible stimuli (tone-band vs. surrounding
  bands). For inaudible stimuli the relevant quantity is the energy delta
  versus the input.
- `respected` is the binary verdict against the threshold for the row's class.

## Headline and gate keys

```json
{
  "masking_respect_score": 1.0,
  "respected_count": 5,
  "stimulus_count": 5,
  "mean_in_band_snr_delta_db": 0.0,
  "mean_inaudible_energy_delta_db": 0.0
}
```

- `masking_respect_score = respected_count / stimulus_count` — the headline.
- `mean_in_band_snr_delta_db` (audible stimuli only) — how much in-band SNR
  drifted vs. the input. Closer to 0 is better.
- `mean_inaudible_energy_delta_db` (inaudible stimuli only) — drift in the
  protected band. Closer to 0 is better; positive numbers mean energy was
  added.

Gate file keys (`gate.yaml → psychoacoustic_masking:`):

- `min_masking_respect_score` — floor on the respect score (0–1).
- `max_inaudible_energy_delta_db` — ceiling on
  `abs(mean_inaudible_energy_delta_db)`.

CLI shortcut: `--min-masking-respect`.

## Adapter contract

Same [`AudioProcessor`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/audio_processor.py)
contract as `ab/fidelity-roundtrip`. Bundled adapters: `passthrough`,
`passthrough-quantize8`, `polarity-flip-right` (none of these regress this
suite — implement your own enhancement / denoise model to actually stress it).

## Scope and caveats

This is a *first-order* psychoacoustic check, not a full Bark-band masking
model. It catches obvious failures (denoiser gates a quiet tone; an
"enhancement" model hallucinates a 1 kHz tone in pure noise). For a full
masking-curve simulation you would want a CB-style auditory front-end; that's
out of scope for the MVP and an explicit non-goal.
