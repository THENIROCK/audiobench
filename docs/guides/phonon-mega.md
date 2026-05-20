# Phonon mega benchmark

The initial mega benchmark curated by **Phonon** (audiobench's company). It runs most laptop-friendly model × suite combinations, documents what was excluded, and flags controversial pairings.

## Run locally

```bash
pip install -e .
# Optional extras for sound-id adapters:
# pip install -e ".[clap]"

chmod +x scripts/run_phonon_mega.sh
./scripts/run_phonon_mega.sh
```

This uses [examples/matrices/phonon-mega.yaml](https://github.com/THENIROCK/audiobench/blob/main/examples/matrices/phonon-mega.yaml) and writes artifacts under `results/phonon-mega/`.

For heavier cells (Whisper medium/large), see [phonon-mega-extended.yaml](https://github.com/THENIROCK/audiobench/blob/main/examples/matrices/phonon-mega-extended.yaml).

## Qwen2-Audio (remote Modal)

Not in the main laptop matrix (no local GPU). Run separately against the Modal endpoint:

```bash
export AUDIOBENCH_QWEN_ENDPOINT=https://thenirock--audiobench-qwen2-qwenserver-web.modal.run
chmod +x scripts/run_phonon_mega_qwen.sh
./scripts/run_phonon_mega_qwen.sh
```

Pre-warm the endpoint once before a full run (cold start can exceed the 120s per-probe timeout). See [qwen2-audio](../models/qwen2-audio.md).

## Publish to the leaderboard

Tag uploads explicitly as Phonon-authored:

```bash
hf auth login
export AUDIOBENCH_LEADERBOARD_DATASET=THENIROCK/audiobench-leaderboard-submissions
export AUDIOBENCH_AUTHOR=Phonon

for f in results/phonon-mega/*.json; do
  audiobench push "$f" --author Phonon --tags phonon-mega-v1
done
```

`--author` sets `authored_by` on the submission (distinct from `submitted_by`, which remains your Hugging Face account). See [Hugging Face leaderboard](hf-leaderboard.md).
You can also use `./scripts/push_phonon_mega.sh` to publish with the shared dataset defaults.

## What is included

| Suite | Models |
| --- | --- |
| `ab/sound-id` | `heuristic-v0`, `heuristic-weak`, `clap-base`, `qwen2-audio-7b` (remote, separate script) |
| `ab/asr-robust` | `whisper-tiny`, `whisper-base`, `whisper-small` |
| `ab/asr-hallucination` | `whisper-tiny`, `whisper-base`, `whisper-small` |

ASR cells use `limit: 20` clips to keep laptop runtimes reasonable. Sound-id cells use `--profile demo-fast`.

## Inapplicable (protocol mismatch)

These combinations are **not** in the matrix because adapters use different protocols:

- Whisper (`transcribe`) cannot run `ab/sound-id` (yes/no probes).
- Sound-id adapters (`answer`) cannot run `ab/asr-robust` or `ab/asr-hallucination`.

## Excluded for compute

| Item | Reason |
| --- | --- |
| `qwen2-audio-7b` (local) | ~16 GB VRAM; use `./scripts/run_phonon_mega_qwen.sh` with Modal instead |
| `whisper-medium`, `whisper-large` | Slow on laptop CPU; in `phonon-mega-extended.yaml` |

## Controversial cells (included with disclosure)

**First-party baselines (`heuristic-v0`, `heuristic-weak`).** Phonon authored both the suite harness and these adapters. They are included as reproducible CPU baselines, not as independent competitors.

**`heuristic-weak`.** Deliberately weak (higher margin threshold + injected jitter). Included only as a diagnostic floor for `audiobench compare`.

**Whisper on `ab/asr-hallucination`.** Highlights a known class of non-speech hallucination behavior. Results are reported for transparency, not as a vendor attack.

**Demo pack overlap.** `heuristic-v0` / `heuristic-weak` are tuned against the same demo pack the suite ships. Treat CLAP and Whisper scores as the more externally meaningful comparisons until external packs land.

## Extend the matrix

Add a cell to `examples/matrices/phonon-mega.yaml`:

```yaml
  - suite: ab/sound-id
    model: my-adapter
    profile: demo-fast
    name: sound-id::my-adapter
```

Then re-run `./scripts/run_phonon_mega.sh`. For new adapters, see [Bring your own model](bring-your-own-model.md).

## Public rankings page

The [Leaderboard](../leaderboard.md) page shows the top five per suite (live from the HF submissions dataset, filtered to `authored_by: Phonon`). Link visitors there for a quick summary; link the HF Space for the full table.
