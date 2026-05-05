from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly


def _as_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def _normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs == 0:
        return audio
    return (audio / max_abs * peak).astype(np.float32)


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = float(np.mean(clean**2)) + 1e-8
    noise_power = float(np.mean(noise**2)) + 1e-8
    target_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return _normalize_peak(clean + noise * scale)


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    spectrum /= np.sqrt(freqs)
    noise = np.fft.irfft(spectrum, n=length)
    return noise.astype(np.float32)


def _cafe_like_noise(length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.standard_normal(length).astype(np.float32)
    b, a = butter(2, [120 / (sample_rate / 2), 3500 / (sample_rate / 2)], btype="band")
    filtered = lfilter(b, a, base).astype(np.float32)
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * (0.2 + rng.random() * 0.3) * t + rng.random() * np.pi)
    chatter = filtered * mod
    return _normalize_peak(chatter, peak=0.9)


def noise_cafe_10db(audio: np.ndarray, sample_rate: int, seed: int) -> tuple[np.ndarray, int]:
    mono = _as_mono(audio)
    rng = np.random.default_rng(seed)
    noise = _cafe_like_noise(len(mono), sample_rate, rng)
    return _mix_at_snr(mono, noise, 10.0), sample_rate


def noise_pink_5db(audio: np.ndarray, sample_rate: int, seed: int) -> tuple[np.ndarray, int]:
    mono = _as_mono(audio)
    rng = np.random.default_rng(seed)
    noise = _pink_noise(len(mono), rng)
    return _mix_at_snr(mono, noise, 5.0), sample_rate


def bandlimited_8k(audio: np.ndarray, sample_rate: int, seed: int) -> tuple[np.ndarray, int]:
    _ = seed
    mono = _as_mono(audio)
    nyq = max(sample_rate / 2.0, 1.0)
    cutoff = min(3400.0 / nyq, 0.99)
    b, a = butter(4, cutoff, btype="low")
    lowpassed = lfilter(b, a, mono).astype(np.float32)
    down = resample_poly(lowpassed, up=1, down=max(sample_rate // 8000, 1))
    up = resample_poly(down, up=max(sample_rate // 8000, 1), down=1)
    up = up[: len(mono)]
    return _normalize_peak(up), sample_rate


def reverb_medium(audio: np.ndarray, sample_rate: int, seed: int) -> tuple[np.ndarray, int]:
    mono = _as_mono(audio)
    rng = np.random.default_rng(seed)
    ir_len = int(sample_rate * 0.4)
    impulse = np.zeros(ir_len, dtype=np.float32)
    impulse[0] = 1.0
    for _ in range(40):
        idx = rng.integers(1, ir_len)
        amp = np.exp(-idx / (sample_rate * 0.2)) * (0.15 + 0.35 * rng.random())
        impulse[idx] += amp
    decay = np.exp(-np.linspace(0, 4, ir_len)).astype(np.float32)
    impulse *= decay
    impulse /= np.sum(np.abs(impulse)) + 1e-8
    wet = fftconvolve(mono, impulse, mode="full")[: len(mono)]
    mixed = 0.75 * mono + 0.25 * wet
    return _normalize_peak(mixed.astype(np.float32)), sample_rate
