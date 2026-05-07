# audiobench MVP CLI

audiobench is a reproducible CLI benchmark for audio ML models. It demonstrates the core idea from [audiobench.dev](https://audiobench.dev): a single clean-set metric hides failure modes, so the benchmark reports performance across realistic perturbations and mixtures.

> **Docs:** <https://thenirock.github.io/audiobench/>

## Suites in this MVP

- `ab/asr-robust` — speech recognition under noise, bandlimiting, and reverb. Default model: Whisper.
- `ab/sound-id` — sound-event identification on mixtures of labeled clips. Default model: a bundled heuristic baseline; real models via CLAP and Qwen2-Audio adapters.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Troubleshooting: `ModuleNotFoundError: No module named 'audiobench'` on macOS

If `audiobench --help` raises `ModuleNotFoundError: No module named 'audiobench'` immediately after `pip install -e .`, this is a known macOS + pip + Python 3.13 `site.py` interaction (Python issue [#127012](https://github.com/python/cpython/issues/127012) / pip issue [#13153](https://github.com/pypa/pip/issues/13153)): pip-installed files inherit a `com.apple.provenance` xattr that carries the `UF_HIDDEN` flag, and Python 3.13's `site.py` skips `.pth` files with that flag, so the editable-install pointer never lands on `sys.path`. Clear the flag on the venv's `site-packages`:

```bash
chflags -R nohidden .venv/lib/python3.13/site-packages
```

### `ab/asr-robust` (speech)

```bash
audiobench run ab/asr-robust --model whisper-tiny
```

Conditions: `clean`, `noise-cafe-10db`, `noise-pink-5db`, `bandlimited-8k`, `reverb-medium`. Reports per-condition WER and weighted mean.

### `ab/sound-id` (sound events)

For each mixture of labeled clips, the model is asked once per candidate label using the bundled prompt set (canonical wording: "Do you hear a {label}?"). The exact wording, version, and any ensemble setting are pinned in the run hash — see the **Prompt protocol** section below.

The benchmark scores how many components of the mixture were correctly identified.

```bash
audiobench run ab/sound-id --model heuristic-v0
```

Conditions are mixture sizes:

- `solo` — N=1 (sanity)
- `pair` — N=2
- `triple` — N=3
- `quad` — N=4

For every `(pack, condition)` row, the benchmark reports:

- **recall** — of the sounds actually in the mixture, what fraction did the model correctly say "yes" to? (1.0 = caught every component; lower = missed some.)
- **precision** — of the times the model said "yes", what fraction were actually present? (1.0 = no false alarms; lower = it claims to hear things that aren't there.)
- **F1** — a single combined score that blends recall and precision; useful when you want one number.
- **FPR** (false-positive rate) — for sounds that are NOT in the mixture (distractors), how often does the model still say "yes"? (0.0 = never hallucinates; higher = it answers "yes" too eagerly.)

Headline number: `components understood: X / Y` — across every mixture, X is how many ground-truth components the model identified out of Y total. **This is the number you'd quote in a tweet.**

## Demo: compare two models on `ab/sound-id`

```bash
audiobench run ab/sound-id --model heuristic-v0    --output results/sound-id-heuristic.json
audiobench run ab/sound-id --model heuristic-weak  --output results/sound-id-weak.json
audiobench compare results/sound-id-heuristic.json results/sound-id-weak.json
```

`compare` dispatches on the suite id in each run JSON, so the same command works for `ab/asr-robust` (lower-WER-wins) and `ab/sound-id` (higher-recall-wins, lower-FPR-wins).

For live presentation, use the demo-fast profile (~30 mixtures, finishes in under 90s on a laptop):

```bash
audiobench run ab/sound-id --profile demo-fast --model heuristic-v0   --output results/demo-heuristic.json
audiobench run ab/sound-id --profile demo-fast --model heuristic-weak --output results/demo-weak.json
audiobench compare results/demo-heuristic.json results/demo-weak.json
```

## Packs

Each `ab/sound-id` run targets one or more **packs**. Each pack defines a label set and source dataset(s).

| Pack | Source | Labels | License |
|---|---|---|---|
| `demo` | Procedural (bundled, no download) | siren, alarm, dog_bark, engine, glass_breaking, baby_cry, coughing, water, vacuum, speech | bundled |
| `core` | FSD50K (single-positive PP filter) | ~80 high-confidence classes from the AudioSet ontology | CC-BY 4.0 / CC0 (user-supplied) |
| `home` | DESED synthetic subset | alarm_bell, cat, dishes, frying, blender, water, speech, vacuum, dog, electric_shaver | open (user-supplied) |
| `cabin` | FSD50K + UrbanSound8K | engine, traffic, baby_cry, music, speech, car_horn, siren, drilling | **non-commercial research** |
| `security` | UrbanSound8K | gun_shot, siren, car_horn, dog_bark, jackhammer | **non-commercial research** |
| `health` | ESC-50 medical subset | coughing, sneezing, breathing, snoring, crying_baby | non-clinical scope |

The `demo` pack runs with no downloads and powers the headline demo. Other packs require user-supplied data at `~/.cache/audiobench/sound_id/<source>/`; see the **Bringing your own data** section below.

```bash
audiobench list-packs
audiobench info ab/sound-id
audiobench info ab/sound-id --pack home
```

## How users create mixtures

Three layers, additive.

**Default** — canned, seeded mixture set per pack. Zero authoring:

```bash
audiobench run ab/sound-id --pack demo --model heuristic-v0
```

**Inline `--mix`** — one mixture per flag, `+`-separated labels:

```bash
audiobench run ab/sound-id --mix "siren+glass_breaking+baby_cry" --model heuristic-v0
audiobench run ab/sound-id --mix "engine+baby_cry" --mix "engine+baby_cry+music" --model heuristic-v0
```

**Recipe file (YAML or JSON)** — repeatable scenarios with per-source dB levels and optional pinned source files:

```yaml
mixtures:
  - name: factory_alarm
    labels: [siren, glass_breaking]
    snr_db: 0

  - name: cabin_baby_over_engine
    label_levels:
      engine: 0
      baby_cry: -3
      vacuum: -6
```

```bash
audiobench run ab/sound-id --recipes scenarios/factory_floor.yaml --model heuristic-v0
```

When `--mix` or `--recipes` is used, results land under a `custom` condition. The run hash includes the canonicalized mixture spec so any custom run is bit-reproducible.

## Prompt protocol

Every `ab/sound-id` run uses a versioned prompt set that lives at [`src/audiobench/data/sound_id/prompts.yaml`](src/audiobench/data/sound_id/prompts.yaml). The version, the parser version, the ensemble size, and a hash of the paraphrase list all feed into `run_hash`, so two runs with different prompt configurations cannot silently be confused.

### Default behavior

By default the suite asks one prompt per probe — the canonical wording "Do you hear a {label}?". The run summary records `prompt_version=yesno-v1, parser=v1, ensemble=off`.

### Inspect or export the bundled prompts

```bash
audiobench prompts show
audiobench prompts export results/my_prompts.yaml
```

`prompts export` writes a starter file you can edit; pass it back with `--prompts`.

### Custom prompts (`--prompts`)

Edit the exported YAML, bump the `version` so old runs aren't confused with the new ones, then point the runner at the file:

```yaml
version: my-clean-room-v1
parser_version: v1
paraphrases:
  - "Do you hear a {label}?"
  - "Listen carefully. Is a {label} present? Reply yes or no."
```

```bash
audiobench run ab/sound-id --prompts results/my_prompts.yaml --model heuristic-v0
```

Schema:

- `version` (required) — opaque label folded into `run_hash`. Any change in wording should bump it.
- `parser_version` (optional, default `v1`) — pinned to the yes/no parser in `audiobench.probes`. Leave it at `v1` unless the parser also changes.
- `paraphrases` (required, ≥ 1) — every entry must contain the literal placeholder `{label}`.

### Prompt ensembles (`--prompt-ensemble N`)

Reduce wording sensitivity by asking N paraphrases per probe and taking a majority vote. The vote is recorded along with each individual paraphrase answer in the run JSON.

```bash
audiobench run ab/sound-id --model qwen2-audio-7b --prompt-ensemble 5
```

`N` must be ≤ the number of paraphrases in the prompts file (5 in the bundled set). The first `N` paraphrases are used in order.

### Comparison guard rails

`audiobench compare` refuses to compare two `ab/sound-id` runs whose `prompt_version`, `parser_version`, or `prompt_ensemble` disagree:

```text
$ audiobench compare results/run-bundled.json results/run-ensemble.json
Invalid value: runs disagree on prompt_ensemble: A=None vs B=3.
Re-run with matching prompts, or pass --allow-mismatched-prompt.
```

Pass `--allow-mismatched-prompt` to override (the comparison header annotates the mismatch).

### Mixture preview

Render a mixture WAV without running probes — useful for demo prep, debugging levels, and authoring recipes:

```bash
audiobench mix preview --labels siren,glass_breaking,baby_cry --output preview.wav
audiobench mix preview --recipes scenarios/factory_floor.yaml --name cabin_baby_over_engine --output cabin.wav
```

## Per-mixture forensic view

```bash
audiobench inspect results/sound-id-heuristic.json --mixture 12
```

```text
mixture 12 (pack=demo, condition=triple)
  ground truth: siren, glass_breaking, dog_bark
  source clips:
    siren           demo://siren@0
    glass_breaking  demo://glass_breaking@0
    dog_bark        demo://dog_bark@0

  model: heuristic-v0
  prompts: version=yesno-v1, parser=v1, ensemble=off (single prompt), source=bundled
  yes responses:
    siren           ✓
    dog_bark        ✓
    glass_breaking  ✗  FALSE NEGATIVE
    chainsaw        ✗  FALSE POSITIVE (distractor)
    car_horn        ✗  (distractor, correct)

  recall    : 2/3 = 0.67
  precision : 2/3 = 0.67
  components understood: 2 of 3
```

When the run was made with `--prompt-ensemble N`, `inspect` also prints a per-paraphrase breakdown showing each rendered prompt and the model's individual yes/no for it.

## Models

`ab/sound-id` ships four model adapters:

- `heuristic-v0` (bundled, CPU) — the strong bundled baseline. See **How the bundled heuristics work** below.
- `heuristic-weak` (bundled, CPU) — a deliberately weaker variant of the same algorithm so `audiobench compare` has something to show out of the box. See **How the bundled heuristics work** below.
- `clap-base` — LAION-CLAP zero-shot. Requires `pip install laion-clap` (lazy import). First run downloads weights.
- `qwen2-audio-7b` — Qwen2-Audio-Instruct via HuggingFace `transformers`. Requires GPU (~16 GB VRAM) locally, or set `AUDIOBENCH_QWEN_ENDPOINT=https://...` to point at a remote inference endpoint. See [docs/models/qwen2-audio.md](docs/models/qwen2-audio.md) for the endpoint contract, a deployable Modal recipe, a free Google Colab + Cloudflared alternative, and Apple Silicon notes.

Add your own model adapter in `src/audiobench/models/` and register it in `src/audiobench/models/registry.py`.

### How the bundled heuristics work

The bundled heuristics aren't ML models — they're a deterministic spectral matcher. They exist so `audiobench` runs end-to-end on a fresh laptop with no GPU, no weight downloads, and no network. The algorithm:

1. **Fingerprint the input audio.** Pad to the next power of two, take an FFT, and bin power into 24 log-spaced frequency bands from 50 Hz to 7.5 kHz. Apply `log1p` to compress dynamic range and L2-normalize. The result is a 24-D unit vector that captures the audio's spectral shape.
2. **Pre-compute one fingerprint per known label** by running the same recipe on the canonical procedural clip for each of the demo pack's 10 labels (`siren`, `engine`, `dog_bark`, …). These reference fingerprints are built once at import time.
3. **Score the probe.** For a question "Do you hear a {label}?", compute the cosine similarity between the input fingerprint and the reference for `{label}` (`target_score`), and the mean cosine similarity to every *other* known label (`baseline`). The decision metric is the **discriminative margin** `margin = target_score − baseline`. Using a margin (rather than the raw similarity) keeps false positives down: in a quad mixture every reference still has decent absolute similarity, but only the components that are actually present beat the rest by a clear margin.
4. **Threshold the margin.** Answer "yes" if `margin >= margin_threshold`, else "no".

The two adapters differ only in two parameters:

| | `margin_threshold` | `noise_amplitude` |
|---|---|---|
| `heuristic-v0` | `0.20` | `0.0` |
| `heuristic-weak` | `0.30` | `0.10` |

- A **higher `margin_threshold`** makes the model more conservative: it answers "yes" only when the target label's match clearly stands out from every other reference. `heuristic-weak`'s `0.30` is well above the typical margin for a true positive in a quad mixture, so it misses many components there — that's the recall hit you'll see in `compare`.
- A **non-zero `noise_amplitude`** adds a small per-decision jitter (`±0.10` here, deterministically derived from a SHA-1 of the label, margin, and audio length, so runs are still reproducible). This lets `heuristic-weak` flip near-threshold decisions, simulating a noisy classifier without breaking the run hash.

Because both heuristics are pure functions of the audio fingerprint and the prompt, they're CPU-only, deterministic, and finish a full `--profile demo-fast` run in well under a second. They're not meant to be competitive with CLAP or Qwen2-Audio — they're meant to make the harness honestly demonstrable on its own.

## Bringing your own data

The `demo` pack runs out of the box. The other packs reference real datasets that you supply:

```text
~/.cache/audiobench/sound_id/
  fsd50k/
    FSD50K.dev_audio/...
  urbansound8k/
    audio/fold1/...
  desed/
    synthetic21_train/soundscapes/...
  esc50/
    audio/...
```

`audiobench info ab/sound-id --pack <name>` prints the expected layout. If files are missing, the suite skips that pack with a clear message rather than failing.

## Other CLI commands

```bash
# Pre-download the Whisper checkpoint so the next `run` doesn't pay for it.
audiobench warmup --model whisper-tiny

# List every suite this build knows about (stable + in-design).
audiobench list

# Print metadata for a suite: clip count, conditions, expected layout.
audiobench info ab/asr-robust

# Run only two ASR conditions and print the run JSON instead of a table.
audiobench run ab/asr-robust --model whisper-tiny --conditions clean,bandlimited-8k --pretty-json

# Print the bundled prompts.yaml (version, parser_version, paraphrase list).
audiobench prompts show

# Copy the bundled prompts.yaml to a path you can edit, then pass it via `--prompts`.
audiobench prompts export results/my_prompts.yaml

# Compare two ab/sound-id runs even if their prompt_version / ensemble settings differ.
# By default `compare` refuses mismatched prompts to keep numbers honest.
audiobench compare results/a.json results/b.json --allow-mismatched-prompt

# Local-only "push" stub: prints a signed payload (suite, revision, run_hash,
# payload_sha256). No network traffic in MVP mode.
audiobench push results/sound-id-heuristic.json --pretty-json
```

## Reproducibility guarantees

- Manifest, mixture, and probe seeds are fixed.
- The mixer is deterministic (peak-normalize, RMS-match, sum).
- The prompt set is versioned and pinned in `run_hash` (`prompt_version`, `parser_version`, `prompt_ensemble`, plus a SHA-256 over the canonicalized paraphrase list).
- Every run writes a JSON artifact with:
  - suite, revision, model, seed, config
  - per-clip / per-mixture hypotheses (with per-paraphrase answers when ensembling)
  - per-condition metrics and weighted mean
  - `run_hash` (SHA-256 over canonicalized run payload, including mixture spec and prompt config)
- `audiobench push` is suite-agnostic and works for both suites; it only reads `suite`, `revision`, `run_hash`.

## Extend this MVP

- Add a model adapter in `src/audiobench/models/` and register it in `models/registry.py`.
- Add a perturbation in `src/audiobench/perturbations.py` for `ab/asr-robust`.
- Add a pack in `src/audiobench/data/sound_id/packs/` for `ab/sound-id`.
- Author a custom prompt set: `audiobench prompts export my_prompts.yaml`, edit, then run with `--prompts my_prompts.yaml`.

## Scope limits

- `ab/sound-id` ships with a procedural `demo` pack so it runs end-to-end without network. Real-data packs (`core`, `home`, `cabin`, `security`, `health`) require user-supplied datasets.
- CLAP and Qwen2-Audio adapters lazy-import heavy deps and are documented but not bundled.
- `push` is local-only and does not send network traffic.
- CPU-first workflow; the bundled heuristic and CLAP models are the CPU-friendly paths.
- English-only data in `ab/asr-robust`.

## Building and deploying the docs

The full docs live at [`docs/`](docs/index.md) and render as a Material-for-MkDocs site. The `[docs]` extra pulls in `mkdocs`, `mkdocs-material`, and `pymdown-extensions`.

**Run locally with hot reload:**

```bash
pip install -e ".[docs]"
mkdocs serve
# open http://127.0.0.1:8000
```

`mkdocs serve` watches `docs/` and `mkdocs.yml`; saving any file rebuilds and refreshes the browser tab.

**Build a static site (output in `site/`, gitignored):**

```bash
mkdocs build --strict
```

`--strict` turns warnings into errors. Useful in CI to catch broken internal links before they ship.

**Deploy to GitHub Pages:**

```bash
mkdocs gh-deploy --force
```

This builds the site and pushes it to a `gh-pages` branch. Then in the GitHub repo settings, under **Pages → Source**, point at the `gh-pages` branch. The site lives at `https://<user>.github.io/audiobench/` (configured in `mkdocs.yml` as `site_url`). For a CI-driven deploy, the [official GitHub Action](https://www.mkdocs.org/user-guide/deploying-your-docs/#github-pages) does the same thing on every push to `main`.
