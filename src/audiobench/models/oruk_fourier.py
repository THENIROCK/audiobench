"""Oruk Fourier 2B adapter for AudioBench ASR suites.

Loads ``oruk/oruk-fourier-2b`` (CC BY-NC 4.0, English Qwen3-ASR 2.04B
derivative) via Hugging Face ``transformers`` (``>=5.24.0``). Heavy ML
dependencies are imported lazily inside ``__post_init__`` so CLI startup
remains fast when this adapter is not selected.

Device and dtype are auto-selected (CUDA bfloat16, Apple MPS bfloat16,
CPU float32) and can be overridden with ``AUDIOBENCH_ORUK_DEVICE`` and
``AUDIOBENCH_ORUK_DTYPE``. The checkpoint repo and revision can be overridden
with ``AUDIOBENCH_ORUK_MODEL_ID`` and ``AUDIOBENCH_ORUK_REVISION``.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from audiobench.models.asr import ASRResult

DEFAULT_MODEL_ID = "oruk/oruk-fourier-2b"
DEFAULT_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
TARGET_SAMPLE_RATE = 16_000
MIN_CHUNK_SAMPLES = 8_000


def _pick_device_and_dtype() -> tuple[str, object]:
    """Pick a ``(device, torch_dtype)`` pair appropriate for this host."""
    import torch

    requested_device = os.environ.get("AUDIOBENCH_ORUK_DEVICE", "").lower().strip()
    if requested_device:
        device = requested_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    requested_dtype = os.environ.get("AUDIOBENCH_ORUK_DTYPE", "").lower().strip()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if requested_dtype:
        if requested_dtype not in dtype_map:
            raise RuntimeError(
                f"unknown AUDIOBENCH_ORUK_DTYPE={requested_dtype!r}; "
                f"valid: {sorted(dtype_map)}"
            )
        dtype = dtype_map[requested_dtype]
    elif device in ("cuda", "mps"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return device, dtype


def _chunk_audio_energy(
    wav: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    min_s: float = 20.0,
    max_s: float = 28.0,
    hard_max_s: float = 30.0,
) -> list[np.ndarray]:
    """Split audio longer than ``hard_max_s`` at minimum-RMS energy points."""
    n = int(wav.shape[0])
    if n == 0:
        return [np.zeros(MIN_CHUNK_SAMPLES, dtype=np.float32)]
    hard_max = int(round(hard_max_s * sr))
    min_n = int(round(min_s * sr))
    max_n = int(round(max_s * sr))
    win_n = max(1, int(round(0.2 * sr)))
    hop_n = max(1, int(round(0.02 * sr)))

    chunks: list[np.ndarray] = []
    pos = 0
    while pos < n:
        if n - pos <= hard_max:
            seg = wav[pos:n]
            pos = n
        else:
            s0 = pos + min_n
            s1 = min(n, pos + max_n)
            if s1 - s0 <= win_n:
                cut = s1
            else:
                band = wav[s0:s1]
                sq = np.empty(band.shape[0] + 1, dtype=np.float64)
                sq[0] = 0.0
                np.cumsum(band.astype(np.float64) ** 2, out=sq[1:])
                starts = np.arange(0, band.shape[0] - win_n + 1, hop_n)
                energies = (sq[starts + win_n] - sq[starts]) / float(win_n)
                cut = s0 + int(starts[int(np.argmin(energies))]) + (win_n // 2)
            seg = wav[pos:cut]
            pos = cut
        if seg.shape[0] < MIN_CHUNK_SAMPLES:
            seg = np.pad(seg, (0, MIN_CHUNK_SAMPLES - seg.shape[0]), mode="constant")
        chunks.append(np.ascontiguousarray(seg, dtype=np.float32))
    return chunks


@dataclass
class OrukFourierTranscriber:
    """Hugging Face Transformers adapter for Oruk Fourier 2B (English ASR)."""

    seed: int = 0
    model_id: str | None = None
    revision: str | None = None
    language: str = "English"
    name: str = "oruk-fourier-2b"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "oruk-fourier-2b requires `transformers>=5.24.0` and `torch>=2.5`. "
                "Install with: pip install 'audiobench[oruk]'"
            ) from exc

        self._torch = torch
        self._seed_everything(self.seed)
        resolved_id = self.model_id or os.environ.get("AUDIOBENCH_ORUK_MODEL_ID", DEFAULT_MODEL_ID)
        resolved_rev = self.revision or os.environ.get("AUDIOBENCH_ORUK_REVISION", DEFAULT_REVISION)
        device, dtype = _pick_device_and_dtype()
        self._device = device
        self._dtype = dtype

        self._processor = AutoProcessor.from_pretrained(resolved_id, revision=resolved_rev)
        if getattr(self._processor, "tokenizer", None) is not None:
            self._processor.tokenizer.padding_side = "left"
        self._model = (
            AutoModelForMultimodalLM.from_pretrained(
                resolved_id,
                revision=resolved_rev,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        t0 = time.perf_counter()
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1 if wav.shape[1] < wav.shape[0] else 0)
        if sample_rate != TARGET_SAMPLE_RATE:
            g = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
            wav = resample_poly(
                wav, TARGET_SAMPLE_RATE // g, int(sample_rate) // g
            ).astype(np.float32, copy=False)

        chunks = _chunk_audio_energy(wav, sr=TARGET_SAMPLE_RATE)
        texts: list[str] = []
        with self._torch.inference_mode():
            for chunk in chunks:
                inputs = self._processor.apply_transcription_request(
                    audio=[chunk],
                    language=[self.language],
                    return_tensors="pt",
                    processor_kwargs={"padding": True},
                ).to(self._device, self._model.dtype)
                output_ids = self._model.generate(**inputs)
                gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                decoded = self._processor.decode(gen_ids, return_format="transcription_only")
                for item in decoded:
                    if item and item.strip():
                        texts.append(item.strip())

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ASRResult(transcript=" ".join(texts).strip(), latency_ms=latency_ms)
