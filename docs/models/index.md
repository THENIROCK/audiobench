# Models

`ab/sound-id` ships four model adapters. They all implement the same protocol: given an audio array, sample rate, and a yes/no prompt, return a free-text answer that the runner parses with `audiobench.probes.parse_yes_no`.

| Adapter | Where it runs | Notes |
|---|---|---|
| `heuristic-v0` | CPU, bundled | Strong bundled baseline. Deterministic spectral matcher; no weights, no network. |
| `heuristic-weak` | CPU, bundled | Deliberately weaker variant of `heuristic-v0`, so `audiobench compare` has something to show out of the box. |
| `clap-base` | CPU, lazy import | LAION-CLAP zero-shot. Requires `pip install laion-clap`. First run downloads weights. |
| [`qwen2-audio-7b`](qwen2-audio.md) | GPU local **or** remote endpoint | Qwen2-Audio-Instruct via HuggingFace `transformers`. ~16 GB VRAM locally; remote API mode runs anywhere. |

`ab/asr-robust` uses a different protocol (transcription, not yes/no) and currently ships `whisper-tiny` and friends from `openai-whisper`.

## How the bundled heuristics work

The bundled heuristics aren't ML models — they're a deterministic spectral matcher. They exist so `audiobench` runs end-to-end on a fresh laptop with no GPU, no weight downloads, and no network. The algorithm:

1. **Fingerprint the input audio.** Pad to the next power of two, take an FFT, and bin power into 24 log-spaced frequency bands from 50 Hz to 7.5 kHz. Apply `log1p` to compress dynamic range and L2-normalize. The result is a 24-D unit vector that captures the audio's spectral shape.
2. **Pre-compute one fingerprint per known label** by running the same recipe on the canonical procedural clip for each of the demo pack's 10 labels. These reference fingerprints are built once at import time.
3. **Score the probe.** For a question `"Do you hear a {label}?"`, compute the cosine similarity between the input fingerprint and the reference for `{label}` (`target_score`), and the mean cosine similarity to every *other* known label (`baseline`). The decision metric is the **discriminative margin** `margin = target_score − baseline`. Using a margin (rather than the raw similarity) keeps false positives down: in a quad mixture every reference still has decent absolute similarity, but only the components that are actually present beat the rest by a clear margin.
4. **Threshold the margin.** Answer "yes" if `margin >= margin_threshold`, else "no".

The two adapters differ only in two parameters:

|  | `margin_threshold` | `noise_amplitude` |
|---|---|---|
| `heuristic-v0` | `0.20` | `0.0` |
| `heuristic-weak` | `0.30` | `0.10` |

A **higher `margin_threshold`** makes the model more conservative: it answers "yes" only when the target label's match clearly stands out from every other reference. `heuristic-weak`'s `0.30` is well above the typical margin for a true positive in a quad mixture, so it misses many components there — that's the recall hit you'll see in `compare`.

A **non-zero `noise_amplitude`** adds a small per-decision jitter (`±0.10` here, deterministically derived from a SHA-1 of the label, margin, and audio length, so runs are still reproducible). This lets `heuristic-weak` flip near-threshold decisions, simulating a noisy classifier without breaking the run hash.

Because both heuristics are pure functions of the audio fingerprint and the prompt, they're CPU-only, deterministic, and finish a full `--profile demo-fast` run in well under a second. They're not meant to be competitive with CLAP or Qwen2-Audio — they're meant to make the harness honestly demonstrable on its own.

## Adding your own adapter

Drop a new module in `src/audiobench/models/`. Implement the [`AudioLLMAdapter`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/audio_llm.py) protocol:

```python
class AudioLLMAdapter(Protocol):
    name: str
    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str: ...
```

Then register it in `src/audiobench/models/registry.py`:

```python
def _make_my_model() -> AudioLLMAdapter:
    from audiobench.models.my_model import MyAdapter
    return MyAdapter()

_FACTORIES["my-model-name"] = _make_my_model
```

Heavy imports (torch, transformers, network calls) should live inside the factory or the adapter constructor, not at module top level. The CLI startup imports the registry eagerly, so a torch import at the top of your adapter module slows down every `audiobench --help`.
