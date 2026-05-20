from __future__ import annotations

import math
import unittest

import numpy as np

from audiobench.signal_metrics import (
    band_energy_db,
    band_snr_db,
    interchannel_correlation,
    k_weighted_loudness_lufs,
    mid_side_round_trip_snr_db,
    mr_stft_log_l1,
    polarity_preservation_score,
    si_sdr_db,
    true_peak_dbtp,
)


SR = 16000


def _sine(freq: float, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * duration_s)) / SR
    return (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


class SiSdrTest(unittest.TestCase):
    def test_identical_signals_high_score(self) -> None:
        x = _sine(440.0, 0.5)
        self.assertGreater(si_sdr_db(x, x), 100.0)

    def test_noisy_signal_lower_score(self) -> None:
        rng = np.random.default_rng(0)
        x = _sine(440.0, 0.5)
        y = x + 0.1 * rng.standard_normal(len(x)).astype(np.float32)
        score = si_sdr_db(x, y)
        self.assertLess(score, 30.0)
        self.assertGreater(score, 0.0)

    def test_scale_invariance(self) -> None:
        x = _sine(440.0, 0.5)
        y = 3.0 * x
        self.assertGreater(si_sdr_db(x, y), 100.0)


class MrStftTest(unittest.TestCase):
    def test_identical_is_zero(self) -> None:
        x = _sine(440.0, 0.5)
        self.assertAlmostEqual(mr_stft_log_l1(x, x), 0.0, places=6)

    def test_increases_with_distortion(self) -> None:
        x = _sine(440.0, 0.5)
        y = _sine(800.0, 0.5)
        # Two different sine frequencies should produce a clearly non-zero loss.
        baseline = mr_stft_log_l1(x, x)
        delta = mr_stft_log_l1(x, y)
        self.assertGreater(delta, baseline + 1e-3)


class TruePeakTest(unittest.TestCase):
    def test_full_scale_is_near_zero_dbtp(self) -> None:
        x = _sine(440.0, 0.5, amplitude=1.0)
        peak = true_peak_dbtp(x)
        self.assertGreaterEqual(peak, -1.0)
        self.assertLess(peak, 1.5)

    def test_quiet_signal_low_peak(self) -> None:
        x = _sine(440.0, 0.5, amplitude=0.001)
        self.assertLess(true_peak_dbtp(x), -40.0)


class LoudnessTest(unittest.TestCase):
    def test_louder_signal_higher_lufs(self) -> None:
        quiet = _sine(1000.0, 1.0, amplitude=0.1)
        loud = _sine(1000.0, 1.0, amplitude=0.5)
        self.assertGreater(
            k_weighted_loudness_lufs(loud, SR),
            k_weighted_loudness_lufs(quiet, SR),
        )

    def test_silence_returns_floor(self) -> None:
        silence = np.zeros(SR, dtype=np.float32)
        self.assertLess(k_weighted_loudness_lufs(silence, SR), -60.0)


class StereoTest(unittest.TestCase):
    def test_correlation_identical_is_one(self) -> None:
        ch = _sine(440.0, 0.5)
        stereo = np.stack([ch, ch.copy()], axis=0)
        self.assertAlmostEqual(interchannel_correlation(stereo), 1.0, places=4)

    def test_correlation_polarity_flip_is_minus_one(self) -> None:
        ch = _sine(440.0, 0.5)
        stereo = np.stack([ch, -ch], axis=0)
        self.assertAlmostEqual(interchannel_correlation(stereo), -1.0, places=4)

    def test_polarity_score_preserved(self) -> None:
        ch = _sine(440.0, 0.5)
        ref = np.stack([ch, ch.copy()], axis=0)
        est = np.stack([ch.copy(), ch.copy()], axis=0)
        self.assertAlmostEqual(polarity_preservation_score(ref, est), 1.0)

    def test_polarity_score_flips_drops(self) -> None:
        ch = _sine(440.0, 0.5)
        ref = np.stack([ch, ch.copy()], axis=0)
        est = np.stack([ch.copy(), -ch.copy()], axis=0)
        self.assertEqual(polarity_preservation_score(ref, est), 0.5)

    def test_ms_round_trip_clean(self) -> None:
        left = _sine(440.0, 0.5)
        right = _sine(660.0, 0.5)
        stereo = np.stack([left, right], axis=0)
        self.assertGreater(mid_side_round_trip_snr_db(stereo), 100.0)


class BandMetricsTest(unittest.TestCase):
    def test_band_energy_picks_target(self) -> None:
        x = _sine(1000.0, 1.0)
        energy_target = band_energy_db(x, SR, 900.0, 1100.0)
        energy_other = band_energy_db(x, SR, 4000.0, 6000.0)
        self.assertGreater(energy_target, energy_other + 20.0)

    def test_band_snr_identical_high(self) -> None:
        x = _sine(1000.0, 1.0)
        self.assertGreater(band_snr_db(x, x, SR, 900.0, 1100.0), 100.0)

    def test_band_snr_drops_when_distorted(self) -> None:
        x = _sine(1000.0, 1.0)
        rng = np.random.default_rng(1)
        y = x + 0.5 * rng.standard_normal(len(x)).astype(np.float32)
        self.assertLess(band_snr_db(x, y, SR, 900.0, 1100.0), 30.0)
