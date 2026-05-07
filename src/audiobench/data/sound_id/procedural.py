"""Deterministic procedural audio generators for the bundled ``demo`` pack.

The demo pack ships no WAV files; clips are synthesized on demand from a fixed
seed so the suite runs end-to-end with no downloads. Each label has a
distinctive spectral signature so the bundled ``heuristic`` model adapter can
distinguish them via simple band-energy ratios.

The signatures are not realistic recordings; they are illustrative. Real-data
packs (FSD50K, DESED, ESC-50, UrbanSound8K) layer on top of this same suite
runner once the user supplies the source files.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import butter, lfilter

DEMO_SAMPLE_RATE = 16000
DEMO_CLIP_SECONDS = 4.0


def _bandpass(noise: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2.0
    low_cut = max(low / nyq, 1e-3)
    high_cut = min(high / nyq, 0.999)
    if high_cut <= low_cut:
        return noise.astype(np.float32)
    b, a = butter(4, [low_cut, high_cut], btype="band")
    return lfilter(b, a, noise).astype(np.float32)


def _normalize(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs == 0:
        return signal.astype(np.float32)
    return (signal / max_abs * peak).astype(np.float32)


def _envelope(length: int, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    attack = int(sr * attack_s)
    release = int(sr * release_s)
    env = np.ones(length, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release > 0:
        env[length - release :] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env


def _make_siren(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    t = np.arange(length, dtype=np.float32) / sr
    sweep = 800.0 + 200.0 * np.sin(2 * np.pi * 0.9 * t + rng.uniform(0, np.pi))
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    sig = np.sin(phase)
    return _normalize(sig.astype(np.float32) * _envelope(length, sr, 0.05, 0.05))


def _make_alarm(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    t = np.arange(length, dtype=np.float32) / sr
    pulse = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 2.5 * t)))
    base = np.sin(2 * np.pi * 1380.0 * t + rng.uniform(0, np.pi))
    sig = base * pulse
    return _normalize(sig.astype(np.float32) * _envelope(length, sr, 0.02, 0.05))


def _make_dog_bark(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    sig = np.zeros(length, dtype=np.float32)
    bark_centers = [int(sr * x) for x in (0.4, 1.1, 1.9, 2.7, 3.4)]
    for center in bark_centers:
        if center >= length:
            continue
        burst_len = int(sr * 0.18)
        end = min(center + burst_len, length)
        n = end - center
        burst = rng.standard_normal(n).astype(np.float32)
        burst = _bandpass(burst, sr, 350.0, 600.0)
        env = np.exp(-np.linspace(0, 6, n, dtype=np.float32))
        sig[center:end] += burst * env
    return _normalize(sig)


def _make_engine(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    t = np.arange(length, dtype=np.float32) / sr
    fundamental = 70.0 + 3.0 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, np.pi))
    phase = 2 * np.pi * np.cumsum(fundamental) / sr
    rumble = 0.7 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    rumble = _bandpass(rumble.astype(np.float32), sr, 50.0, 220.0)
    rough = _bandpass(rng.standard_normal(length).astype(np.float32), sr, 60.0, 200.0)
    sig = 0.9 * rumble + 0.2 * rough
    return _normalize(sig.astype(np.float32) * _envelope(length, sr, 0.1, 0.1))


def _make_glass_breaking(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    sig = np.zeros(length, dtype=np.float32)
    crash = int(sr * 0.6)
    burst = rng.standard_normal(crash).astype(np.float32)
    high = _bandpass(burst, sr, 4000.0, 7500.0)
    env = np.exp(-np.linspace(0, 10, crash, dtype=np.float32))
    sig[: len(high)] = high * env
    tail_start = int(sr * 0.65)
    tail_len = int(sr * 0.8)
    if tail_start + tail_len < length:
        tail = rng.standard_normal(tail_len).astype(np.float32)
        tail = _bandpass(tail, sr, 4500.0, 7500.0)
        tail_env = np.exp(-np.linspace(0, 8, tail_len, dtype=np.float32))
        sig[tail_start : tail_start + tail_len] += 0.5 * tail * tail_env
    return _normalize(sig)


def _make_baby_cry(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    t = np.arange(length, dtype=np.float32) / sr
    pitch = 1900.0 + 250.0 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, np.pi))
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    fundamental = np.sin(phase)
    cry_env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    sig = fundamental.astype(np.float32) * cry_env.astype(np.float32)
    return _normalize(sig * _envelope(length, sr, 0.08, 0.1))


def _make_coughing(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    sig = np.zeros(length, dtype=np.float32)
    cough_starts = [int(sr * x) for x in (0.3, 1.5, 2.8)]
    for start in cough_starts:
        if start >= length:
            continue
        cough_len = int(sr * 0.35)
        end = min(start + cough_len, length)
        n = end - start
        body = rng.standard_normal(n).astype(np.float32)
        body = _bandpass(body, sr, 220.0, 380.0)
        attack = int(n * 0.1)
        env = np.ones(n, dtype=np.float32)
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        env *= np.exp(-np.linspace(0, 5, n, dtype=np.float32))
        sig[start:end] += body * env
    return _normalize(sig)


def _make_water(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    base = rng.standard_normal(length).astype(np.float32)
    filtered = _bandpass(base, sr, 2400.0, 3600.0)
    t = np.arange(length, dtype=np.float32) / sr
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, np.pi))
    return _normalize(filtered * mod.astype(np.float32))


def _make_vacuum(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    base = rng.standard_normal(length).astype(np.float32)
    filtered = _bandpass(base, sr, 250.0, 800.0)
    t = np.arange(length, dtype=np.float32) / sr
    hum = 0.4 * np.sin(2 * np.pi * 380.0 * t + rng.uniform(0, np.pi))
    sig = filtered + hum.astype(np.float32)
    return _normalize(sig * _envelope(length, sr, 0.15, 0.15))


def _make_speech(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    length = int(sr * dur)
    base = rng.standard_normal(length).astype(np.float32)
    voiced = _bandpass(base, sr, 1100.0, 2300.0)
    t = np.arange(length, dtype=np.float32) / sr
    syllables = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4.5 * t + rng.uniform(0, np.pi)))
    smooth = np.convolve(syllables, np.ones(int(sr * 0.04)) / max(int(sr * 0.04), 1), mode="same")
    return _normalize(voiced * smooth.astype(np.float32))


GENERATORS: dict[str, Callable[[int, float, np.random.Generator], np.ndarray]] = {
    "siren": _make_siren,
    "alarm": _make_alarm,
    "dog_bark": _make_dog_bark,
    "engine": _make_engine,
    "glass_breaking": _make_glass_breaking,
    "baby_cry": _make_baby_cry,
    "coughing": _make_coughing,
    "water": _make_water,
    "vacuum": _make_vacuum,
    "speech": _make_speech,
}

DEMO_LABELS: list[str] = list(GENERATORS.keys())


def synthesize(label: str, *, variant: int = 0, sr: int = DEMO_SAMPLE_RATE, duration: float = DEMO_CLIP_SECONDS) -> np.ndarray:
    if label not in GENERATORS:
        raise KeyError(f"no procedural generator for label: {label}")
    seed = (hash(("audiobench-demo", label, int(variant))) & 0xFFFFFFFF) ^ 0xDEADBEEF
    rng = np.random.default_rng(seed)
    return GENERATORS[label](sr, duration, rng)


# Spectral fingerprints used by the bundled heuristic adapters. Each tuple is
# (low_hz, high_hz). The heuristic checks whether a mixture has elevated energy
# inside the band relative to a wide-band reference; a stronger ratio means the
# label is more likely to be present.
LABEL_BANDS: dict[str, tuple[float, float]] = {
    "siren": (600.0, 1050.0),
    "alarm": (1300.0, 1450.0),
    "dog_bark": (350.0, 600.0),
    "engine": (50.0, 220.0),
    "glass_breaking": (4000.0, 7500.0),
    "baby_cry": (1700.0, 2200.0),
    "coughing": (220.0, 380.0),
    "water": (2400.0, 3600.0),
    "vacuum": (250.0, 800.0),
    "speech": (1100.0, 2300.0),
}
