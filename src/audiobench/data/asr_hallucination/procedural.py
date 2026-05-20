"""Deterministic non-speech clip synthesis for ab/asr-hallucination."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from scipy.signal import butter, lfilter

from audiobench.hashing import sha256_text


def _normalize_peak(audio: np.ndarray, *, peak: float = 0.85) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= 0:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    if freqs.size > 1:
        freqs[0] = freqs[1]
    spectrum /= np.sqrt(np.maximum(freqs, 1e-6))
    noise = np.fft.irfft(spectrum, n=length)
    return noise.astype(np.float32)


def _bandpass(noise: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    nyquist = max(sample_rate / 2.0, 1.0)
    low_cut = max(low / nyquist, 1e-4)
    high_cut = min(high / nyquist, 0.999)
    if high_cut <= low_cut:
        return noise.astype(np.float32)
    b, a = butter(2, [low_cut, high_cut], btype="band")
    return lfilter(b, a, noise).astype(np.float32)


def _silence_floor(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    floor = rng.standard_normal(length).astype(np.float32) * 0.0008
    return floor


def _white_noise(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    return _normalize_peak(rng.standard_normal(length).astype(np.float32) * 0.2)


def _pink_noise_clip(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    return _normalize_peak(_pink_noise(length, rng) * 0.25)


def _cafe_noise(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    base = rng.standard_normal(length).astype(np.float32)
    chatter = _bandpass(base, sample_rate, 120.0, 3200.0)
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * (0.2 + 0.25 * rng.random()) * t + rng.random() * np.pi)
    return _normalize_peak(chatter * modulation.astype(np.float32) * 0.3)


def _room_tone(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    hum = 0.02 * np.sin(2 * np.pi * 60.0 * t + rng.random() * np.pi)
    hiss = _bandpass(rng.standard_normal(length).astype(np.float32), sample_rate, 3000.0, 7000.0) * 0.03
    return _normalize_peak(hum.astype(np.float32) + hiss.astype(np.float32), peak=0.25)


def _music_bed(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(sample_rate * duration_s))
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    root = 220.0 + 8.0 * np.sin(2 * np.pi * 0.15 * t + rng.random() * np.pi)
    third = root * 5.0 / 4.0
    fifth = root * 3.0 / 2.0
    lead = 0.35 * np.sin(2 * np.pi * root * t)
    lead += 0.22 * np.sin(2 * np.pi * third * t + 0.3)
    lead += 0.18 * np.sin(2 * np.pi * fifth * t + 0.6)
    pulse = 0.6 + 0.4 * np.sin(2 * np.pi * 1.2 * t + rng.random() * np.pi)
    hiss = _bandpass(rng.standard_normal(length).astype(np.float32), sample_rate, 6000.0, 7800.0) * 0.02
    return _normalize_peak((lead * pulse + hiss).astype(np.float32) * 0.8)


GENERATORS: dict[str, Callable[[int, float, np.random.Generator], np.ndarray]] = {
    "silence_floor": _silence_floor,
    "white_noise": _white_noise,
    "pink_noise": _pink_noise_clip,
    "cafe_noise": _cafe_noise,
    "room_tone": _room_tone,
    "music_bed": _music_bed,
}


def _clip_seed(clip: Mapping[str, Any], *, global_seed: int) -> int:
    seed_token = "|".join(
        [
            "ab/asr-hallucination",
            str(clip.get("id", "")),
            str(clip.get("generator", "")),
            str(clip.get("seed", "")),
            str(global_seed),
        ]
    )
    return int(sha256_text(seed_token)[:8], 16)


def synthesize_clip(
    clip: Mapping[str, Any],
    *,
    sample_rate: int,
    global_seed: int = 0,
) -> np.ndarray:
    """Synthesize one manifest clip deterministically."""

    generator = str(clip.get("generator", ""))
    if generator not in GENERATORS:
        raise KeyError(f"unknown asr hallucination generator: {generator}")
    duration_s = float(clip.get("duration_s", 2.0))
    rng = np.random.default_rng(_clip_seed(clip, global_seed=global_seed))
    audio = GENERATORS[generator](sample_rate, duration_s, rng)
    return np.asarray(audio, dtype=np.float32)
