from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import numpy as np

from audiobench.models.asr import ASRResult, normalize_asr_response
from audiobench.models.oruk_fourier import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    MIN_CHUNK_SAMPLES,
    TARGET_SAMPLE_RATE,
    _chunk_audio_energy,
)


class _DummyAdapter:
    def __init__(self, seed: int) -> None:
        self.name = "oruk-fourier-2b"
        self.seed = seed

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        _ = (audio, sample_rate)
        return ASRResult(transcript="hello world", latency_ms=12.5)


class OrukFourierRegistryTest(unittest.TestCase):
    def test_oruk_fourier_registered_in_asr_registry(self) -> None:
        registry = importlib.reload(importlib.import_module("audiobench.models.asr_registry"))
        models = registry.list_models()
        self.assertIn("oruk-fourier-2b", models)
        self.assertIn("whisper-*", models)

    def test_make_model_dispatches_to_oruk_fourier(self) -> None:
        registry = importlib.reload(importlib.import_module("audiobench.models.asr_registry"))
        with patch(
            "audiobench.models.oruk_fourier.OrukFourierTranscriber",
            side_effect=lambda seed: _DummyAdapter(seed),
        ):
            adapter = registry.make_model("oruk-fourier-2b", seed=42)
        self.assertEqual(adapter.name, "oruk-fourier-2b")
        result = normalize_asr_response(adapter.transcribe(np.zeros(1600, dtype=np.float32), 16000))
        self.assertEqual(result.transcript, "hello world")
        self.assertEqual(result.latency_ms, 12.5)

    def test_default_hf_coordinates_and_chunking(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID, "oruk/oruk-fourier-2b")
        self.assertEqual(len(DEFAULT_REVISION), 40)
        # Empty audio pads safely to minimum chunk length
        empty_chunks = _chunk_audio_energy(np.zeros(0, dtype=np.float32))
        self.assertEqual(len(empty_chunks), 1)
        self.assertEqual(empty_chunks[0].shape[0], MIN_CHUNK_SAMPLES)
        # Long 65s audio splits into <= 30s chunks
        long_audio = np.zeros(65 * TARGET_SAMPLE_RATE, dtype=np.float32)
        chunks = _chunk_audio_energy(long_audio)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(chunk.shape[0], 30 * TARGET_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
