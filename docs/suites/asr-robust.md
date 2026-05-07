# `ab/asr-robust`

Speech recognition under common acoustic perturbations.

```bash
audiobench run ab/asr-robust --model whisper-tiny
```

## Conditions

The suite runs every clip through five conditions:

| Condition | What it does |
|---|---|
| `clean` | Untouched reference. |
| `noise-cafe-10db` | Mixes in cafe babble at +10 dB SNR. |
| `noise-pink-5db` | Mixes in pink noise at +5 dB SNR. |
| `bandlimited-8k` | Low-passes to 8 kHz then resamples back, mimicking a narrowband channel. |
| `reverb-medium` | Convolves with a medium-room impulse response. |

Reports per-condition WER plus a weighted mean over all conditions. The mean weights each condition equally, not by clip count, so adding clips to a single condition doesn't shift the headline.

## Useful flags

Run only specific conditions and dump machine-readable JSON:

```bash
audiobench run ab/asr-robust --model whisper-tiny \
  --conditions clean,bandlimited-8k --pretty-json
```

Pre-download the Whisper checkpoint so the next `run` doesn't pay for it:

```bash
audiobench warmup --model whisper-tiny
```

## Adding a perturbation

Open `src/audiobench/perturbations.py` and add a new entry. Each perturbation is a `(name, callable)` pair where `callable(audio, sample_rate, seed) -> audio`. Update the suite manifest to include it. The seed is folded into `run_hash`, so two runs with the same perturbation list and the same model produce the same hash.

## Scope

English-only data. The reference clips are short read-speech sentences; this isn't a long-form streaming benchmark.
