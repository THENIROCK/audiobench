"""Signal-level metrics shared by the Phase 3 audio benchmarks.

These deliberately stay close to standard recipes (BS.1770-ish K-weighting,
SI-SDR, oversampled true peak, multi-resolution STFT log-magnitude L1) and
are implemented in pure NumPy/SciPy so they run on a laptop with no extras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal


_EPS = 1e-12


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=arr.shape.index(min(arr.shape)))
    raise ValueError(f"unsupported audio shape {arr.shape!r}")


def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB (higher is better)."""
    ref = _to_mono(reference)
    est = _to_mono(estimate)
    ref, est = _align_length(ref, est)
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = float(np.dot(ref, ref))
    if ref_energy < _EPS:
        return 0.0
    alpha = float(np.dot(est, ref)) / ref_energy
    target = alpha * ref
    noise = est - target
    target_energy = float(np.dot(target, target))
    noise_energy = float(np.dot(noise, noise))
    if noise_energy < _EPS:
        return 120.0
    if target_energy < _EPS:
        return -120.0
    return 10.0 * math.log10(target_energy / noise_energy)


def mr_stft_log_l1(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    frame_sizes: tuple[int, ...] = (256, 1024, 4096),
) -> float:
    """Multi-resolution log-magnitude STFT L1 (lower is better)."""
    ref = _to_mono(reference)
    est = _to_mono(estimate)
    ref, est = _align_length(ref, est)
    losses: list[float] = []
    for n in frame_sizes:
        nperseg = min(n, len(ref))
        if nperseg < 16:
            continue
        noverlap = nperseg // 2
        _, _, sx = scipy_signal.stft(ref, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
        _, _, sy = scipy_signal.stft(est, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
        magnitude_ref = np.log1p(np.abs(sx))
        magnitude_est = np.log1p(np.abs(sy))
        losses.append(float(np.mean(np.abs(magnitude_ref - magnitude_est))))
    if not losses:
        return 0.0
    return float(np.mean(losses))


def true_peak_dbtp(audio: np.ndarray, *, oversample: int = 4) -> float:
    """4x oversampled true peak in dBTP (0 dBTP = full scale)."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        channels = [arr]
    else:
        channels = [arr[:, i] if arr.shape[0] > arr.shape[1] else arr[i] for i in range(min(arr.shape))]
    peaks: list[float] = []
    for ch in channels:
        if len(ch) == 0:
            continue
        upsampled = scipy_signal.resample_poly(ch, oversample, 1)
        peaks.append(float(np.max(np.abs(upsampled))))
    if not peaks:
        return -120.0
    peak = max(peaks)
    if peak < _EPS:
        return -120.0
    return 20.0 * math.log10(peak)


def _k_weighting_filter(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 K-weighting filter as a cascaded biquad pair (b, a)."""
    # Stage 1: pre-filter (shelving high-shelf at ~1.5 kHz, +4 dB)
    f0 = 1681.97
    g = 3.999843
    q = 0.7071752
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (g / 20.0)
    vb = vh ** 0.499666
    a0 = 1.0 + k / q + k * k
    b0_1 = (vh + vb * k / q + k * k) / a0
    b1_1 = 2.0 * (k * k - vh) / a0
    b2_1 = (vh - vb * k / q + k * k) / a0
    a1_1 = 2.0 * (k * k - 1.0) / a0
    a2_1 = (1.0 - k / q + k * k) / a0
    # Stage 2: RLB highpass at ~38 Hz
    f0 = 38.13547
    q = 0.5003270
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    b0_2 = 1.0 / a0
    b1_2 = -2.0 / a0
    b2_2 = 1.0 / a0
    a1_2 = 2.0 * (k * k - 1.0) / a0
    a2_2 = (1.0 - k / q + k * k) / a0
    b1 = np.array([b0_1, b1_1, b2_1])
    a1 = np.array([1.0, a1_1, a2_1])
    b2 = np.array([b0_2, b1_2, b2_2])
    a2 = np.array([1.0, a1_2, a2_2])
    # Convolve cascade
    b = np.convolve(b1, b2)
    a = np.convolve(a1, a2)
    return b, a


def k_weighted_loudness_lufs(audio: np.ndarray, sr: int) -> float:
    """Approximate BS.1770 integrated loudness (LUFS, mono fold-down).

    Skips the BS.1770 gating step (sufficient for short benchmark clips
    where gating would not remove any blocks anyway).
    """
    mono = _to_mono(audio)
    if len(mono) < int(0.4 * sr):
        return -120.0
    b, a = _k_weighting_filter(sr)
    filtered = scipy_signal.lfilter(b, a, mono)
    mean_square = float(np.mean(filtered * filtered))
    if mean_square < _EPS:
        return -120.0
    return -0.691 + 10.0 * math.log10(mean_square)


def split_stereo(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (L, R) if ``audio`` is stereo, else None."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 2 and arr.shape[1] >= 2:
        return arr[0], arr[1]
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    return None


def interchannel_correlation(audio: np.ndarray) -> float:
    """Pearson correlation between L and R; 1.0 = identical, -1.0 = polarity flip."""
    pair = split_stereo(audio)
    if pair is None:
        return 1.0
    left, right = pair
    left, right = _align_length(left, right)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < _EPS:
        return 0.0
    return float(np.dot(left, right) / denom)


def polarity_preservation_score(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """1.0 if processed channels keep the reference's L/R correlation sign, 0.0 otherwise.

    Computed across channels: for each channel index, take the per-sample sign
    of the cross-correlation with the reference channel and average. A polarity
    flip on one channel drops the score by ``1/num_channels``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        return 0.0
    if ref.ndim == 1:
        ref = ref[None, :]
        est = est[None, :]
    channels = ref.shape[0]
    matches = 0
    for ch in range(channels):
        r = ref[ch] - ref[ch].mean()
        e = est[ch] - est[ch].mean()
        denom = float(np.linalg.norm(r) * np.linalg.norm(e))
        if denom < _EPS:
            matches += 1
            continue
        corr = float(np.dot(r, e) / denom)
        matches += 1 if corr >= 0 else 0
    return matches / float(channels)


def mid_side_round_trip_snr_db(audio: np.ndarray) -> float:
    """SNR (dB) for L/R → M/S → L/R reconstruction; high for any sane processor."""
    pair = split_stereo(audio)
    if pair is None:
        return 120.0
    left, right = pair
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    recon_left = mid + side
    recon_right = mid - side
    err = np.concatenate([left - recon_left, right - recon_right])
    sig_energy = float(np.mean(left * left + right * right))
    err_energy = float(np.mean(err * err))
    if err_energy < _EPS:
        return 120.0
    if sig_energy < _EPS:
        return -120.0
    return 10.0 * math.log10(sig_energy / err_energy)


@dataclass(frozen=True)
class BandEnergy:
    band_hz: tuple[float, float]
    energy_db: float


def band_energy_db(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Mean power-spectral-density energy in [low_hz, high_hz] expressed in dB."""
    mono = _to_mono(audio)
    if len(mono) < 16:
        return -120.0
    freqs, pxx = scipy_signal.welch(mono, fs=sr, nperseg=min(2048, len(mono)))
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return -120.0
    energy = float(np.mean(pxx[mask]))
    if energy < _EPS:
        return -120.0
    return 10.0 * math.log10(energy)


def band_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
) -> float:
    """In-band SNR between estimate and reference (dB)."""
    ref = _to_mono(reference)
    est = _to_mono(estimate)
    ref, est = _align_length(ref, est)
    sos = scipy_signal.butter(
        4,
        [max(low_hz, 1e-3), min(high_hz, sr * 0.49)],
        btype="band",
        fs=sr,
        output="sos",
    )
    ref_band = scipy_signal.sosfiltfilt(sos, ref)
    est_band = scipy_signal.sosfiltfilt(sos, est)
    err = est_band - ref_band
    ref_energy = float(np.dot(ref_band, ref_band))
    err_energy = float(np.dot(err, err))
    if err_energy < _EPS:
        return 120.0
    if ref_energy < _EPS:
        return -120.0
    return 10.0 * math.log10(ref_energy / err_energy)
