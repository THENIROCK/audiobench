"""Deterministic mixer for ab/sound-id.

Each mixture is built from N source signals with the same sample rate. The
mixer:

1. Length-equalizes by truncating to the shortest source (or padding shorter
   ones with zeros, depending on ``equalize_mode``).
2. Peak-normalizes each source.
3. Applies a per-source dB level (optional ``label_levels``); when not
   supplied, sources sum at equal RMS at the configured ``snr_db`` between
   sources (default 0 dB, i.e. equal loudness).
4. Sums and peak-normalizes the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixSource:
    label: str
    audio: np.ndarray
    sample_rate: int


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32)


def _peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs == 0:
        return audio.astype(np.float32, copy=False)
    return (audio / max_abs * peak).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def mix_sources(
    sources: list[MixSource],
    *,
    snr_db: float = 0.0,
    label_levels: dict[str, float] | None = None,
    equalize_mode: str = "truncate",
) -> tuple[np.ndarray, int]:
    """Mix labeled audio sources deterministically.

    Args:
        sources: list of (label, audio, sample_rate). All sample rates must
            match.
        snr_db: per-source level offset relative to the first source. The
            default (0 dB) sums sources at equal RMS.
        label_levels: optional explicit dB offset per label, overriding
            ``snr_db`` for that source. Useful for recipes like
            ``baby_cry: -3``.
        equalize_mode: ``"truncate"`` (default) cuts to the shortest source;
            ``"pad"`` pads shorter sources with zeros.

    Returns:
        (mixture, sample_rate).
    """
    if not sources:
        raise ValueError("mix_sources requires at least one source")

    sample_rates = {s.sample_rate for s in sources}
    if len(sample_rates) != 1:
        raise ValueError(f"all sources must share sample rate; got {sample_rates}")
    sr = sources[0].sample_rate

    monos = [_to_mono(s.audio) for s in sources]

    if equalize_mode == "truncate":
        length = min(len(m) for m in monos)
        monos = [m[:length] for m in monos]
    elif equalize_mode == "pad":
        length = max(len(m) for m in monos)
        padded = []
        for m in monos:
            if len(m) < length:
                pad = np.zeros(length - len(m), dtype=np.float32)
                m = np.concatenate([m, pad])
            padded.append(m)
        monos = padded
    else:
        raise ValueError(f"unknown equalize_mode: {equalize_mode}")

    if length == 0:
        raise ValueError("mix_sources: zero-length sources after equalization")

    peaked = [_peak_normalize(m) for m in monos]

    target_rms = _rms(peaked[0])
    if target_rms <= 0:
        target_rms = 0.1

    out = np.zeros(length, dtype=np.float32)
    for source, mono in zip(sources, peaked):
        if label_levels and source.label in label_levels:
            level_db = float(label_levels[source.label])
        else:
            level_db = float(snr_db) if source.label != sources[0].label else 0.0
        scaled_target_rms = target_rms * _db_to_amp(level_db)
        current_rms = _rms(mono)
        scale = scaled_target_rms / current_rms if current_rms > 0 else 1.0
        out = out + (mono * scale).astype(np.float32)

    return _peak_normalize(out), sr
