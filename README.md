# audiobench

audiobench is a reproducible CLI benchmark for audio ML models.
It emphasizes failure modes that clean-set scores hide, and records auditable run artifacts (`run_hash`, per-condition metrics, and suite-specific evidence such as hallucination findings).

- Docs site: [https://thenirock.github.io/audiobench/](https://thenirock.github.io/audiobench/)
- Repository: [https://github.com/THENIROCK/audiobench](https://github.com/THENIROCK/audiobench)

## Install

Python 3.10+. A virtual environment keeps dependencies isolated; activate it in each new shell.

### From PyPI

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install audiobench
audiobench --help
```

Optional extras:

```bash
pip install "audiobench[gui]"    # local Gradio UI (audiobench --gui)
pip install "audiobench[clap]"   # LAION-CLAP adapter for ab/sound-id
pip install "audiobench[qwen]"   # local Qwen2-Audio adapter
```

### From source (development)

For hacking on the repo, clone and install in editable mode so changes under `src/` apply immediately:

```bash
git clone https://github.com/THENIROCK/audiobench.git
cd audiobench
python -m venv .venv
source .venv/bin/activate
pip install -e .
audiobench --help
```

Use `pip install -e ".[gui]"` (and `.[clap]`, `.[qwen]`) for the same extras from a checkout.

If you hit `ModuleNotFoundError: No module named 'audiobench'` on macOS + Python 3.13 after an editable install, see the workaround in `docs/quickstart.md`.

## Local GUI (optional)

For structuring multi-suite testing sessions (matrix YAML) and browsing run
results by session, there is a small local Gradio app:

```bash
pip install "audiobench[gui]"   # or: pip install -e ".[gui]" from a clone
audiobench --gui
```

It opens a local browser tab with a Test Builder (compose `matrix.yaml`) and a
Results view (sessions = directories under `results/` containing
`summary.json`). The other CLI verbs (`run`, `compare`, `inspect`, `gate`,
`push`) are available as simple forms, or you can stay in the terminal — the
GUI is a thin wrapper, not a replacement. See
[docs/guides/gui.md](docs/guides/gui.md) for details.

## Core CLI commands

```bash
# Discover suites and adapters
audiobench list
audiobench list-models
audiobench info ab/asr-hallucination

# Run a suite and write JSON artifact
audiobench run ab/asr-hallucination --model whisper-tiny --output results/hallucination.json

# Run several (suite, model) cells at once and aggregate results
audiobench run-matrix \
    --suite ab/sound-id --model heuristic-v0 --model heuristic-weak \
    --profile demo-fast --output-dir results/matrix

# Compare two run artifacts (same suite)
audiobench compare results/run-a.json results/run-b.json

# Inspect a single record (per-clip for ASR, per-mixture for sound-id)
audiobench inspect results/hallucination.json --clip 1
audiobench inspect results/sound-id.json --mixture 1

# Gate a run artifact against thresholds (exits non-zero on failure)
audiobench gate results/hallucination.json --max-hallucination-rate 0.1

# Publish a run artifact to the leaderboard dataset
hf auth login
audiobench push results/hallucination.json
```

## Shipped suites

Task suites (model = transcriber / sound-event identifier):

| Suite | Purpose | Headline metric |
| --- | --- | --- |
| `ab/asr-robust` | Speech recognition under perturbations (`clean`, noise, bandlimit, reverb) | `weighted_mean_wer` (lower is better) |
| `ab/asr-hallucination` | Non-speech ASR hallucination stress test (`silence`, `music`, `noise`) with statistical findings | `weighted_hallucination_rate` (lower is better) + finding validation status |
| `ab/sound-id` | Sound-event identification on labeled mixtures | `weighted_recall` (higher is better) + `weighted_fpr` (lower is better) |

Signal suites (model = `AudioProcessor` adapter that takes `(audio, sr)` and returns processed audio):

| Suite | Purpose | Headline metric |
| --- | --- | --- |
| `ab/fidelity-roundtrip` | Audio fidelity under round-trip (procedural sweep, noise, impulses, low-level + near-clip tones; identity / band-limit / +3 dB conditions) | `weighted_si_sdr_db` (higher is better), `max_true_peak_dbtp` (lower is better) |
| `ab/psychoacoustic-masking` | Whether the processor preserves audible tones and leaves masked tones masked (tone-in-noise fixtures) | `masking_respect_score` (higher is better) |
| `ab/phase-coherence` | Stereo polarity, inter-channel correlation, M/S round-trip, sub-sample delay preservation | `phase_coherence_score`, `mean_polarity_score` (both higher is better) |

Temporal task suites (frame-level scoring with IoU-matched events or Hungarian-aligned speakers):

| Suite | Purpose | Headline metric |
| --- | --- | --- |
| `ab/sed-urban` | Sound event detection on labeled urban-noise soundscapes (sirens, dog barks, alarms, engines, glass) | `event_f1_iou50` (higher is better), `segment_f1_1s` (higher is better) |
| `ab/diarization-cw` | Speaker diarization on procedural conversations (DER decomposed into miss / FA / confusion with 0.25 s collar) | `der` (lower is better), `mean_speaker_count_error` (lower is better) |

Run each suite quickly:

```bash
audiobench run ab/asr-robust --model whisper-tiny
audiobench run ab/asr-hallucination --model whisper-tiny
audiobench run ab/sound-id --model heuristic-v0

audiobench run ab/fidelity-roundtrip      --model passthrough
audiobench run ab/psychoacoustic-masking  --model passthrough
audiobench run ab/phase-coherence         --model passthrough

audiobench run ab/sed-urban       --model oracle-sed
audiobench run ab/diarization-cw  --model oracle-diarization
```

The signal suites ship three reference adapters out of the box:

- `passthrough` — identity (the upper bound; should pass every check).
- `passthrough-quantize8` — 8-bit quantizer (visibly degrades fidelity, useful as a regression demo).
- `polarity-flip-right` — flips the right channel polarity (fails phase coherence on purpose).

Implement your own by writing an adapter that fulfils
[`AudioProcessor`](src/audiobench/models/audio_processor.py) (one
`process(audio, sample_rate) -> (audio_out, sample_rate_out)` method) and
either registering it in
[`models/signal_registry.py`](src/audiobench/models/signal_registry.py) or
publishing it as an `audiobench.signal_models` entry point.

The temporal suites use task-specific adapter contracts:

- **SED** — implement
  [`SEDAdapter`](src/audiobench/models/sed.py) with
  `detect(audio, sample_rate) -> list[{"label", "start_s", "end_s"}]` and
  register via [`models/sed_registry.py`](src/audiobench/models/sed_registry.py)
  or `audiobench.sed_models` entry points. Bundled adapters:
  `oracle-sed` (sanity upper bound), `oracle-sed-jittered` (boundary-jittered
  regression demo), `null-sed` (worst case).
- **Diarization** — implement
  [`DiarizationAdapter`](src/audiobench/models/diarization.py) with
  `diarize(audio, sample_rate) -> list[{"speaker_id", "start_s", "end_s"}]`
  and register via
  [`models/diarization_registry.py`](src/audiobench/models/diarization_registry.py)
  or `audiobench.diarization_models` entry points. Bundled adapters:
  `oracle-diarization`, `merged-diarization` (all speakers collapsed —
  confusion regression), `single-speaker` (one big turn — FA + confusion).

Both oracle baselines accept a `set_oracle_hint(ground_truth)` call from the
suite runner so they can answer the procedural manifest perfectly; production
adapters simply ignore that hint and rely on the audio.

## Findings and validation flow (`ab/asr-hallucination`)

Each run includes ranked findings with:

- effect size (`effect_size`)
- bootstrap confidence interval (`ci_lower`, `ci_upper`)
- Benjamini-Hochberg corrected p-value (`adjusted_p_value`)
- validation status (`validated`, `candidate`, `rejected`)

Useful flow:

```bash
audiobench run ab/asr-hallucination --model whisper-tiny --output results/hallucination-whisper.json
audiobench run ab/asr-hallucination --model my-asr-model --output results/hallucination-my-model.json
audiobench compare results/hallucination-whisper.json results/hallucination-my-model.json
```

For publishable-claim policy and reproducibility checklist, see `docs/guides/repro-launch-flow.md`.

## Leaderboard publish flow

`audiobench push` uploads a run artifact to:

`submissions/<suite-with-/-replaced-by-__>/<run_hash>.json`

If `--repo` is omitted, push auto-targets:

`<your-username>/audiobench-leaderboard-submissions`

Example:

```bash
hf auth login
audiobench push results/hallucination-my-model.json --pretty-json
```

Optional flags:

- `--repo my-org/audiobench-leaderboard-submissions`
- `--space <org-or-user>/<space-name>`
- `--notes "..." --tags "cpu,baseline"`
- `--dry-run` to validate payload without upload

The Space app scaffold for a hosted leaderboard is in `spaces/leaderboard/`.

## Models and adapters

List adapters:

```bash
audiobench list-models
audiobench list-models --suite ab/asr-hallucination
```

Built-in `ab/sound-id` adapters include `heuristic-v0`, `heuristic-weak`, `clap-base`, and `qwen2-audio-7b`.
Install optional extras when needed:

```bash
pip install "audiobench[clap]"
pip install "audiobench[qwen]"
```

Bring-your-own-model guide: `docs/guides/bring-your-own-model.md`.

## Methodology and docs

- Quickstart: `docs/quickstart.md`
- Suites overview: `docs/suites/index.md`
- `ab/asr-hallucination` reference: `docs/suites/asr-hallucination.md`
- Reproducibility guarantees: `docs/reference/reproducibility.md`
- Repro launch policy for findings: `docs/guides/repro-launch-flow.md`
- Leaderboard integration guide: `docs/guides/hf-leaderboard.md`

## Extras for `ab/sound-id`

```bash
# Pack discovery and availability
audiobench list-packs
audiobench info ab/sound-id --pack demo

# Prompt controls
audiobench prompts show
audiobench prompts export results/my_prompts.yaml

# Mixture authoring
audiobench run ab/sound-id --mix "siren+dog_bark" --model heuristic-v0
audiobench run ab/sound-id --recipes examples/scenarios/factory_floor.yaml --model heuristic-v0
audiobench mix preview --recipes examples/scenarios/factory_floor.yaml --name factory_alarm --output preview.wav

# Per-mixture debugging
audiobench inspect results/run.json --mixture 1
```

## CI integration

`audiobench gate` evaluates a run JSON against thresholds and exits non-zero
when any check fails, so it can fail PRs on regressions. Thresholds can come
from CLI flags or a YAML/JSON file with suite-aware sections.

```bash
# Inline thresholds (sound-id headline metrics)
audiobench gate results/sound-id.json --min-recall 0.6 --max-fpr 0.1

# Inline thresholds (ASR robust, including per-condition caps)
audiobench gate results/asr-robust.json \
    --max-wer 30 \
    --max-wer-condition clean=10 \
    --max-wer-condition noise-cafe-10db=40

# File-based thresholds
audiobench gate results/run.json --thresholds gate.yaml --json
```

Example `gate.yaml`:

```yaml
asr_robust:
  max_weighted_mean_wer: 30.0
  max_wer:
    clean: 10.0
    noise-cafe-10db: 40.0
asr_hallucination:
  max_weighted_hallucination_rate: 0.10
  max_non_speech_hallucination_rate: 0.15
sound_id:
  min_weighted_recall: 0.60
  max_weighted_fpr: 0.10
  min_components_understood: 20
fidelity_roundtrip:
  min_weighted_si_sdr_db: 80.0
  max_true_peak_dbtp: 6.0
  max_mean_loudness_delta_lu: 1.0
psychoacoustic_masking:
  min_masking_respect_score: 0.99
  max_inaudible_energy_delta_db: 3.0
phase_coherence:
  min_phase_coherence_score: 0.99
  min_mean_polarity_score: 0.99
sed_urban:
  min_event_f1: 0.6
  min_segment_f1: 0.6
  min_event_recall: 0.5
diarization_cw:
  max_der: 0.25
  max_speaker_count_error: 0.5
  max_miss_rate: 0.15
  max_false_alarm_rate: 0.15
```

Inline shortcuts cover the most common signal and task checks too:
`--min-si-sdr`, `--max-true-peak`, `--min-masking-respect`,
`--min-phase-coherence`, `--min-polarity`,
`--min-event-f1`, `--min-segment-f1`,
`--max-der`, `--max-speaker-count-error`.

`gate` also accepts `--junit out.xml` to emit a JUnit-style report (one
testcase per check), which GitHub Actions, GitLab, Jenkins, and other CI
systems can parse natively.

The repo ships a reference workflow at `.github/workflows/ci.yml` that runs
the test suite and then a smoke `audiobench run-matrix` with `--gate` and
`--junit` against `ab/sound-id` on the bundled demo pack.

## Orchestration: `run-matrix`

`audiobench run-matrix` runs many `(suite, model)` cells in one invocation,
writes a per-cell run JSON, and emits an aggregated summary. The same matrix
can be built and run from the **Test Builder** tab of `audiobench --gui`. With `--gate`
each cell is also threshold-checked; with `--junit` the result becomes a
single XML report consumable by CI dashboards. The command exits non-zero if
any cell errors out or any gate check fails.

```bash
# Cartesian over repeated --suite / --model flags
audiobench run-matrix \
    --suite ab/sound-id \
    --model heuristic-v0 --model heuristic-weak \
    --profile demo-fast \
    --output-dir results/matrix \
    --gate gate.yaml \
    --junit results/matrix/junit.xml

# Or fully declarative via a matrix YAML
audiobench run-matrix --matrix matrix.yaml
```

Example `matrix.yaml`:

```yaml
output_dir: results/matrix
seed: 1337
cells:
  - suite: ab/sound-id
    model: heuristic-v0
    profile: demo-fast
  - suite: ab/sound-id
    model: heuristic-weak
    profile: demo-fast
  - suite: ab/asr-robust
    model: tiny
    conditions: [clean]
    limit: 2
gate:
  sound_id:
    min_weighted_recall: 0.4
  asr_robust:
    max_weighted_mean_wer: 80.0
```

The aggregated summary lands at `<output-dir>/summary.json` with one entry
per cell (run hash, headline metrics, gate result, duration) so downstream
tooling can render scorecards without re-reading each run artifact.

## Inspect: per-record forensics

`audiobench inspect` opens a single record from a run JSON. The flag depends
on the suite:

```bash
# ab/asr-robust and ab/asr-hallucination: per clip
audiobench inspect results/asr-robust.json --clip 1
audiobench inspect results/asr-hallucination.json --clip 3

# ab/sound-id: per mixture
audiobench inspect results/sound-id.json --mixture 12
```

ASR clip view shows the reference, every condition's hypothesis, per-clip WER,
latency, and flags (`empty`, `hallucination`, `error`). Sound-id mixture view
shows ground-truth components, per-label yes/no probes, and (when ensembling)
per-paraphrase breakdowns.
