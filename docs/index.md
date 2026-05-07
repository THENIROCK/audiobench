---
hide:
  - navigation
  - toc
---

# audiobench

**A reproducible CLI benchmark for audio ML models.**

A single clean-set metric hides failure modes. audiobench reports performance across realistic perturbations and mixtures — so you find out where a model actually breaks, not just how it scores on the easy slice.

[Get started](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/THENIROCK/audiobench){ .md-button }

---

## What's in the MVP

<div class="grid cards" markdown>

-   :material-microphone-message:{ .lg .middle } **`ab/asr-robust`**

    ---

    Speech recognition under noise, bandlimiting, and reverb. Per-condition WER plus a weighted mean. Default model: Whisper.

    [:octicons-arrow-right-24: Suite reference](suites/asr-robust.md)

-   :material-music-note-eighth:{ .lg .middle } **`ab/sound-id`**

    ---

    Sound-event identification on labeled mixtures. Reports recall, precision, F1, and false-positive rate per mixture size. Default model: a bundled CPU heuristic.

    [:octicons-arrow-right-24: Suite reference](suites/sound-id.md)

-   :material-cube-outline:{ .lg .middle } **Model adapters**

    ---

    Bundled heuristics, LAION-CLAP zero-shot, and Qwen2-Audio-7B-Instruct (local GPU or remote endpoint).

    [:octicons-arrow-right-24: Models](models/index.md)

-   :material-shield-check:{ .lg .middle } **Reproducibility built in**

    ---

    Manifest, mixture, probe, and prompt seeds are pinned. Every run writes a JSON artifact with a `run_hash`.

    [:octicons-arrow-right-24: Reproducibility guarantees](reference/reproducibility.md)

</div>

---

## In one command

```bash
pip install -e .
audiobench run ab/sound-id --model heuristic-v0
```

That gets you a full `ab/sound-id` run on the bundled `demo` pack, no downloads, no GPU. From there:

```bash
audiobench run ab/sound-id --profile demo-fast --model heuristic-v0   --output results/demo-heuristic.json
audiobench run ab/sound-id --profile demo-fast --model heuristic-weak --output results/demo-weak.json
audiobench compare results/demo-heuristic.json results/demo-weak.json
```

The `compare` command dispatches on the suite id baked into each run JSON, so the same call works for `ab/asr-robust` (lower-WER-wins) and `ab/sound-id` (higher-recall-wins, lower-FPR-wins).

---

## Where to go next

- **New here?** Start with the [quickstart](quickstart.md).
- **Running on a real dataset?** See [packs and bring-your-own-data](suites/sound-id.md#packs).
- **Trying Qwen2-Audio?** The [qwen2-audio guide](models/qwen2-audio.md) has a Modal recipe and a free Colab fallback for laptops without a GPU.
- **Adding a model?** [Models overview](models/index.md) covers the adapter protocol.
