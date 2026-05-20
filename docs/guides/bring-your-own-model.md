# Bring your own model

If you are evaluating your own model, the shortest path is:

1. Implement the adapter protocol.
2. Register it (in-repo or via Python entry points).
3. Run `audiobench run` with your adapter id.

## `ab/sound-id` adapter contract

Your adapter needs:

- `name: str`
- `answer(audio: np.ndarray, sample_rate: int, prompt: str) -> str`

`audiobench` converts your free-text answer to yes/no with `parse_yes_no`, so return plain "yes" / "no" style strings.

```python
import numpy as np


class MySoundIdAdapter:
    name = "my-sound-model"

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        # Replace with your inference path.
        _ = (audio, sample_rate, prompt)
        return "no"
```

## Registering your adapter

### Option A: register inside this repo

Add your adapter in `src/audiobench/models/`, then add a factory in `src/audiobench/models/registry.py`.

### Option B: ship a plugin package (no fork)

`audiobench` discovers third-party adapters via Python entry points in the `audiobench.models` group.

```toml
[project.entry-points."audiobench.models"]
my-sound-model = "my_pkg.adapters:make_adapter"
```

The factory must be callable as `make_adapter() -> AudioLLMAdapter`.

## `ab/asr-robust` adapter contract

Your ASR adapter needs:

- `name: str`
- `transcribe(audio: np.ndarray, sample_rate: int) -> str | dict`

```python
import numpy as np


class MyAsrAdapter:
    name = "my-asr-model"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        _ = (audio, sample_rate)
        return {
            "transcript": "",
            "latency_ms": 42.0,   # optional
            "cost_usd": 0.001,    # optional
            "error": None,        # optional
        }
```

Returning a plain string is still supported for backwards compatibility.

For plugins, use the `audiobench.asr_models` entry-point group:

```toml
[project.entry-points."audiobench.asr_models"]
my-asr-model = "my_pkg.asr:make_asr_adapter"
```

The factory must be callable as `make_asr_adapter(seed: int) -> ASRAdapter`.

## Run and inspect

```bash
# Show discoverable adapters.
audiobench list-models

# Run your sound-id adapter.
audiobench run ab/sound-id --model my-sound-model

# Run your ASR adapter.
audiobench run ab/asr-robust --model my-asr-model
```

If you are using Whisper directly, `ab/asr-robust` still accepts checkpoint names such as `whisper-tiny`.
