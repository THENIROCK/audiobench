"""Audio processor adapter contract for Phase 3 signal-level suites.

A processor takes ``(audio, sample_rate)`` and returns ``(audio_out, sample_rate_out)``.
``audio`` is a NumPy array, mono (1-D) or stereo (2 × N), float32 in [-1, 1].
Adapters are expected to be deterministic and self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AudioProcessor(Protocol):
    """Protocol for any model under test in a signal-level suite."""

    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        ...


@dataclass
class IdentityAdapter:
    """Pass-through processor — useful as a baseline / sanity check."""

    name: str = "passthrough"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return np.asarray(audio, dtype=np.float32).copy(), int(sample_rate)


@dataclass
class QuantizeAdapter:
    """Reduce bit depth, then expand back. Demonstrably degrades fidelity."""

    name: str = "passthrough-quantize8"
    bits: int = 8

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(audio, dtype=np.float64)
        peak = max(float(np.max(np.abs(arr))), 1e-9)
        scale = (2 ** (self.bits - 1)) - 1
        quantized = np.round(arr / peak * scale) / scale * peak
        return quantized.astype(np.float32), int(sample_rate)


@dataclass
class PolarityFlipAdapter:
    """Flip the polarity of the right channel — known phase regression."""

    name: str = "polarity-flip-right"

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(audio, dtype=np.float32).copy()
        if arr.ndim == 2:
            if arr.shape[0] == 2:
                arr[1] = -arr[1]
            elif arr.shape[1] == 2:
                arr[:, 1] = -arr[:, 1]
        return arr, int(sample_rate)


def make_passthrough() -> IdentityAdapter:
    return IdentityAdapter()


def make_quantize8() -> QuantizeAdapter:
    return QuantizeAdapter(name="passthrough-quantize8", bits=8)


def make_polarity_flip_right() -> PolarityFlipAdapter:
    return PolarityFlipAdapter()
