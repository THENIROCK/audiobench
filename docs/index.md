---
hide:
  - navigation
  - toc
---

# audiobench

**A reproducible CLI benchmark for audio ML models.**

A single clean-set metric hides failure modes. audiobench reports performance across realistic perturbations and mixtures — so you find out where a model actually breaks, not just how it scores on the easy slice.

[Get started](quickstart.md){ .md-button .md-button--primary }
[View rankings](leaderboard.md){ .md-button }
[View on GitHub](https://github.com/THENIROCK/audiobench){ .md-button }

---

## What's in the MVP

<div class="grid cards" markdown>

-   :material-microphone-message:{ .lg .middle } **`ab/asr-robust`**

    ---

    Speech recognition under noise, bandlimiting, and reverb. Per-condition WER plus a weighted mean. Default model: Whisper.

    [:octicons-arrow-right-24: Suite reference](suites/asr-robust.md)

-   :material-alert-outline:{ .lg .middle } **`ab/asr-hallucination`**

    ---

    Non-speech hallucination benchmark with ranked findings, bootstrap CIs, and holdout validation status.

    [:octicons-arrow-right-24: Suite reference](suites/asr-hallucination.md)

-   :material-music-note-eighth:{ .lg .middle } **`ab/sound-id`**

    ---

    Sound-event identification on labeled mixtures. Reports recall, precision, F1, and false-positive rate per mixture size. Default model: a bundled CPU heuristic.

    [:octicons-arrow-right-24: Suite reference](suites/sound-id.md)

-   :material-waveform:{ .lg .middle } **Signal suites**

    ---

    Reference-aware fidelity, psychoacoustic masking, and stereo phase checks for any `AudioProcessor` adapter (codec, DSP chain, plug-in, neural enhancement).

    [:octicons-arrow-right-24: Fidelity](suites/fidelity-roundtrip.md) ·
    [Psychoacoustics](suites/psychoacoustic-masking.md) ·
    [Phase](suites/phase-coherence.md)

-   :material-timeline-clock-outline:{ .lg .middle } **Temporal task suites**

    ---

    Frame-level event detection (IoU-matched F1) and speaker diarization (NIST DER with Hungarian alignment and a 0.25 s collar).

    [:octicons-arrow-right-24: SED](suites/sed-urban.md) ·
    [Diarization](suites/diarization-cw.md)

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
pip install audiobench
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

## Benchmark your own model

If your goal is to evaluate your model, start from this flow:

1. Implement the adapter protocol (`answer(...)` for `ab/sound-id`, `transcribe(...)` for `ab/asr-robust`).
2. Register it in-repo, or expose it as a Python entry point.
3. Run with your adapter id.

```bash
audiobench list-models
audiobench run ab/sound-id --model my-sound-model
audiobench run ab/asr-robust --model my-asr-model
```

The complete adapter and plugin setup lives in [Bring your own model](guides/bring-your-own-model.md).

---

## Where to go next

- **New here?** Start with the [quickstart](quickstart.md).
- **Running on a real dataset?** See [packs and bring-your-own-data](suites/sound-id.md#packs).
- **Trying Qwen2-Audio?** The [qwen2-audio guide](models/qwen2-audio.md) has a Modal recipe and a free Colab fallback for laptops without a GPU.
- **Adding a model?** [Models overview](models/index.md) covers the adapter protocol.
- **Publishing scores?** [Hugging Face leaderboard integration](guides/hf-leaderboard.md) shows the Space + `audiobench push` flow.

---

## Telemetry and privacy

**Default:** audiobench does not phone home. Model adapters you choose (Whisper,
Qwen, etc.) may download weights from Hugging Face — that is separate from
audiobench telemetry.

**First run (interactive terminal):** you may see a one-time prompt asking
whether to share anonymous usage stats (command name, suite, adapter,
duration, success/failure). Default is **no**. Nothing is sent unless you
accept or set `AUDIOBENCH_TELEMETRY=1`.

**What we never collect:** audio, transcripts, run JSON, file paths, IP
addresses, or credentials. See the full schema in
[Telemetry reference](reference/telemetry.md).

**Opt out anytime:**

```bash
export AUDIOBENCH_TELEMETRY=0
# or delete ~/.config/audiobench/consent.json
```

**Docs site:** cookieless [GoatCounter](https://www.goatcounter.com/) page
views (aggregate country + referrer only). Honours Do Not Track.

**PyPI:** download counts by Python/OS; country breakdown via public BigQuery
when we run the snapshot job. No install referrer from PyPI.

**Public dashboard:** [Analytics](analytics/index.html) merges PyPI, GitHub,
mentions (HN / Reddit / Bluesky), and opt-in CLI aggregates.

[Analytics setup guide](guides/analytics-setup.md) ·
[Take the 3-question survey](https://tally.so/r/dW8edA){ .md-button }
