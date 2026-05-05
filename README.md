# audiobench MVP CLI

audiobench is a reproducible CLI benchmark for audio ML models. This MVP focuses on one suite, `ab/asr-robust`, and demonstrates the core idea from [audiobench.dev](https://audiobench.dev): a single clean-set metric hides failure modes, so the benchmark reports performance across realistic perturbations.

## What this MVP includes

- One implemented suite: `ab/asr-robust`
- 10 bundled speech clips with transcripts
- 5 evaluation conditions:
  - `clean`
  - `noise-cafe-10db`
  - `noise-pink-5db`
  - `bandlimited-8k`
  - `reverb-medium`
- Deterministic run hash (`manifest + config + hypotheses`)
- Local `push` stub for signed payload preview

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the benchmark:

```bash
audiobench run ab/asr-robust --model whisper-tiny
```

Example output shape:

```text
ab/asr-robust · tiny · 10 clips × 5 conditions · seed=1337

condition             WER   Δ vs clean
clean                4.80   —
noise-cafe-10db     12.30  +7.50
noise-pink-5db       9.10  +4.30
bandlimited-8k      18.70 +13.90
reverb-medium        8.40  +3.60
weighted mean       10.70

run hash: ab-asr-robust@0.1.0 · 3f9ce1b2…e1b2
wrote: results/run-3f9ce1b2.json
```

> Note: first run downloads Whisper model weights.

## CLI commands

```bash
audiobench list
audiobench info ab/asr-robust
audiobench run ab/asr-robust --model whisper-tiny
audiobench push results/run-xxxxxxxx.json
```

Optional run flags:

- `--seed`: deterministic seed (default `1337`)
- `--limit`: evaluate only first N clips for quick checks
- `--output`: explicit JSON output path
- `--json`: print machine-readable JSON to stdout

## Reproducibility guarantees

- Manifest and perturbation seeds are fixed.
- Decoding is deterministic (`temperature=0`, seeded runtime).
- Every run writes a JSON artifact with:
  - suite/revision/model/seed/config
  - per-clip hypotheses
  - per-condition WER and weighted mean
  - `run_hash` (SHA-256 over canonicalized run payload)

## YC demo flow

Use the script:

```bash
./examples/demo.sh
```

It runs:

1. `audiobench list`
2. `audiobench info ab/asr-robust`
3. `audiobench run ... --model whisper-tiny`
4. `audiobench run ... --model whisper-base`
5. `audiobench push` on latest result

## Extend this MVP

- Add a model adapter in `src/audiobench/models/`
- Add a perturbation in `src/audiobench/perturbations.py`
- Register conditions in `src/audiobench/suites/asr_robust.py`

## Scope limits in this MVP

- Only `ab/asr-robust` is implemented.
- `push` is local-only and does not send network traffic.
- CPU-first workflow; no GPU optimization path.
- English-only demo data.
