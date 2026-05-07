# Quickstart

## Install

audiobench is an editable Python package. Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Confirm the CLI is on your `PATH`:

```bash
audiobench --help
```

??? warning "macOS + Python 3.13: `ModuleNotFoundError: No module named 'audiobench'`"
    If `audiobench --help` raises `ModuleNotFoundError: No module named 'audiobench'` immediately after `pip install -e .`, this is a known macOS + pip + Python 3.13 `site.py` interaction (Python issue [#127012](https://github.com/python/cpython/issues/127012) / pip issue [#13153](https://github.com/pypa/pip/issues/13153)).

    pip-installed files inherit a `com.apple.provenance` xattr that carries the `UF_HIDDEN` flag, and Python 3.13's `site.py` skips `.pth` files with that flag, so the editable-install pointer never lands on `sys.path`. Clear the flag on the venv's `site-packages`:

    ```bash
    chflags -R nohidden .venv/lib/python3.13/site-packages
    ```

??? danger "macOS + iCloud Drive: hung Python imports"
    If your project sits under `~/Documents` (or any iCloud Drive folder), macOS may evict `.venv` files under memory pressure (`compressed,dataless` xattr). Fresh Python processes then hang on `read()` of evicted `.pyc` files while iCloud tries to fetch them back. Move the project somewhere outside iCloud:

    ```bash
    mkdir -p ~/code
    cp -R ~/Documents/audiobench ~/code/
    cd ~/code/audiobench
    rm -rf .venv && python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

    Verify with `find .venv -type f -flags +dataless | wc -l` (should print `0`).

## First run: `ab/sound-id` on the demo pack

The `demo` pack runs end-to-end with no downloads and no GPU. Good first sanity check:

```bash
audiobench run ab/sound-id --model heuristic-v0
```

For each mixture, the model is asked once per candidate label using the bundled prompt set (canonical wording: `"Do you hear a {label}?"`). The benchmark scores how many components of the mixture were correctly identified.

You'll see four conditions:

- `solo` — N=1 (sanity)
- `pair` — N=2
- `triple` — N=3
- `quad` — N=4

Each `(pack, condition)` row reports:

- **recall** — of the sounds actually in the mixture, what fraction did the model correctly say "yes" to? (`1.0` = caught every component; lower = missed some.)
- **precision** — of the times the model said "yes", what fraction were actually present? (`1.0` = no false alarms; lower = it claims to hear things that aren't there.)
- **F1** — a single combined score blending recall and precision; useful when you want one number.
- **FPR** (false-positive rate) — for sounds that are NOT in the mixture (distractors), how often does the model still say "yes"? (`0.0` = never hallucinates; higher = it answers "yes" too eagerly.)

Headline number: `components understood: X / Y` — across every mixture, X is how many ground-truth components the model identified out of Y total. **This is the number you'd quote in a tweet.**

## First run: `ab/asr-robust`

```bash
audiobench run ab/asr-robust --model whisper-tiny
```

Conditions: `clean`, `noise-cafe-10db`, `noise-pink-5db`, `bandlimited-8k`, `reverb-medium`. Reports per-condition WER and a weighted mean.

## Compare two models

```bash
audiobench run ab/sound-id --model heuristic-v0    --output results/sound-id-heuristic.json
audiobench run ab/sound-id --model heuristic-weak  --output results/sound-id-weak.json
audiobench compare results/sound-id-heuristic.json results/sound-id-weak.json
```

`compare` dispatches on the suite id in each run JSON, so the same command works for `ab/asr-robust` (lower-WER-wins) and `ab/sound-id` (higher-recall-wins, lower-FPR-wins).

For a live-presentation-friendly profile (~30 mixtures, finishes in under 90 s on a laptop):

```bash
audiobench run ab/sound-id --profile demo-fast --model heuristic-v0   --output results/demo-heuristic.json
audiobench run ab/sound-id --profile demo-fast --model heuristic-weak --output results/demo-weak.json
audiobench compare results/demo-heuristic.json results/demo-weak.json
```

## What's next

- [`ab/sound-id`](suites/sound-id.md) — full reference, including packs, custom mixtures, and recipe files.
- [`ab/asr-robust`](suites/asr-robust.md) — perturbation list and WER reporting.
- [Models](models/index.md) — bundled adapters and how to plug in your own.
- [qwen2-audio-7b without a local GPU](models/qwen2-audio.md) — Modal recipe, Colab fallback.
