# Changelog

All notable changes to **audiobench** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-07

First public release on PyPI. (Version `0.1.0` was tagged in git as part
of pre-release work but was never published to PyPI; `0.1.1` is the first
version users can `pip install`.)

### Added

- `ab/asr-robust` suite: speech recognition under `clean`, `noise-cafe-10db`,
  `noise-pink-5db`, `bandlimited-8k`, and `reverb-medium` conditions. Reports
  per-condition WER and a weighted mean. Default model: Whisper.
- `ab/sound-id` suite: sound-event identification on labeled mixtures, with
  `solo` / `pair` / `triple` / `quad` mixture-size conditions. Reports
  recall, precision, F1, and false-positive rate per `(pack, condition)`,
  plus a headline "components understood: X / Y" number.
- Bundled `demo` pack runs end-to-end on a fresh laptop with no GPU, no
  weight downloads, and no network. Five additional packs (`core`, `home`,
  `cabin`, `security`, `health`) reference user-supplied datasets at
  `~/.cache/audiobench/sound_id/`.
- Four model adapters for `ab/sound-id`:
  - `heuristic-v0` and `heuristic-weak`: deterministic spectral matchers
    bundled with the package; no GPU, no downloads.
  - `clap-base`: LAION-CLAP zero-shot. Lazy-imported.
  - `qwen2-audio-7b`: Qwen2-Audio-Instruct via local `transformers` or a
    remote endpoint via `AUDIOBENCH_QWEN_ENDPOINT`. Lazy-imported.
- Versioned prompt protocol pinned in `run_hash` (`prompt_version`,
  `parser_version`, `prompt_ensemble`, plus a SHA-256 over the canonicalized
  paraphrase list). `audiobench compare` refuses mismatched prompts unless
  `--allow-mismatched-prompt` is passed.
- Recipe-driven mixture authoring: bundled defaults, inline `--mix
  "label1+label2"`, and YAML/JSON recipe files with per-source dB levels.
- Mixture preview command: `audiobench mix preview` writes a WAV without
  running probes.
- Forensic per-mixture view: `audiobench inspect <run.json> --mixture N`.
- `audiobench compare` for both suites; `audiobench push` writes a signed
  local payload (no network in MVP mode).
- Reproducibility guarantees: fixed manifest/mixture/probe seeds,
  deterministic mixer, and a `run_hash` SHA-256 over the canonicalized run
  payload (mixture spec + prompt config) written into every run JSON.
- Optional installs:
  - `pip install "audiobench[clap]"` — adds the LAION-CLAP zero-shot adapter.
  - `pip install "audiobench[qwen]"` — adds local Qwen2-Audio via
    `transformers` + `torch`.
  - `pip install "audiobench[docs]"` — MkDocs Material toolchain.
  - `pip install "audiobench[dev]"` — `pytest`, `build`, `twine`.

### Notes

- Python 3.10+ supported; tested on 3.13.
- macOS users may hit `ModuleNotFoundError` immediately after
  `pip install -e .` due to a known macOS + pip + Python 3.13 `site.py`
  interaction (Python issue
  [#127012](https://github.com/python/cpython/issues/127012) /
  pip issue [#13153](https://github.com/pypa/pip/issues/13153)). Fix:
  `chflags -R nohidden .venv/lib/python3.13/site-packages`.

[Unreleased]: https://github.com/THENIROCK/audiobench/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/THENIROCK/audiobench/releases/tag/v0.1.1
