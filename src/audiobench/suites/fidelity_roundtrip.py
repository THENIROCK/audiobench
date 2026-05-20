"""ab/fidelity-roundtrip — audio fidelity under processor round-trip.

The processor under test sees a small set of procedural signals (sine sweep,
white noise, sparse impulses, low-level tone, high-headroom tone), each
optionally pre-perturbed by a condition. We compare the processor output to
the suite-provided reference (the condition's input) and report SI-SDR,
multi-resolution STFT loss, true-peak headroom delta, and integrated
loudness drift.

Headline:
- ``weighted_si_sdr_db`` (higher is better)
- ``max_true_peak_dbtp`` (lower is better; > 0 dBTP = inter-sample clip)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from audiobench.hashing import manifest_hash, run_hash
from audiobench.models.audio_processor import AudioProcessor
from audiobench.models.signal_registry import make_model
from audiobench.signal_metrics import (
    k_weighted_loudness_lufs,
    mr_stft_log_l1,
    si_sdr_db,
    true_peak_dbtp,
)


SUITE_ID = "ab/fidelity-roundtrip"
SUITE_REVISION = "0.1.0"
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Procedural fixtures
# ---------------------------------------------------------------------------


def _sine(freq: float, duration_s: float, *, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def _sine_sweep(f_start: float, f_end: float, duration_s: float) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    k = (f_end / f_start) ** (1.0 / max(duration_s, 1e-6))
    phase = 2.0 * math.pi * f_start * (k ** t - 1.0) / math.log(k)
    return (0.5 * np.sin(phase)).astype(np.float32)


def _white_noise(duration_s: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(int(SAMPLE_RATE * duration_s)) * 0.25
    return audio.astype(np.float32)


def _impulse_train(duration_s: float, *, period_s: float = 0.2) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    audio = np.zeros(n, dtype=np.float32)
    step = max(1, int(SAMPLE_RATE * period_s))
    audio[::step] = 0.9
    return audio


def _high_headroom_sine(freq: float, duration_s: float) -> np.ndarray:
    # Peak right below 0 dBFS — sensitive to any gain or rounding error.
    return _sine(freq, duration_s, amplitude=0.97)


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    description: str
    factory: Callable[[], np.ndarray]


STIMULI: tuple[Stimulus, ...] = (
    Stimulus("sine-440", "Pure 440 Hz tone at -6 dBFS", lambda: _sine(440.0, 1.0)),
    Stimulus("sine-sweep", "Log sweep 50 Hz → 7.5 kHz", lambda: _sine_sweep(50.0, 7500.0, 2.0)),
    Stimulus("white-noise", "Gaussian white noise, seed=7", lambda: _white_noise(1.0, seed=7)),
    Stimulus("impulse-train", "Sparse impulses (every 200 ms)", lambda: _impulse_train(1.0)),
    Stimulus("low-level-sine", "1 kHz tone at -40 dBFS", lambda: _sine(1000.0, 1.0, amplitude=0.01)),
    Stimulus("high-headroom-sine", "880 Hz tone at -0.26 dBFS", lambda: _high_headroom_sine(880.0, 1.0)),
)


# ---------------------------------------------------------------------------
# Conditions (input perturbations applied BEFORE the processor sees the audio)
# ---------------------------------------------------------------------------


def _identity(audio: np.ndarray) -> np.ndarray:
    return audio.copy()


def _bandlimit_8k(audio: np.ndarray) -> np.ndarray:
    from scipy import signal as scipy_signal

    sos = scipy_signal.butter(8, 4000.0, btype="low", fs=SAMPLE_RATE, output="sos")
    return scipy_signal.sosfiltfilt(sos, audio).astype(np.float32)


def _gain_plus_3db(audio: np.ndarray) -> np.ndarray:
    # +3 dB; clips of clip near full-scale (intentional headroom stress).
    return np.clip(audio * (10.0 ** (3.0 / 20.0)), -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Condition:
    name: str
    transform: Callable[[np.ndarray], np.ndarray]


CONDITIONS: tuple[Condition, ...] = (
    Condition("identity", _identity),
    Condition("bandlimit-8k", _bandlimit_8k),
    Condition("gain-+3db", _gain_plus_3db),
)


# ---------------------------------------------------------------------------
# Manifest (used for hashing)
# ---------------------------------------------------------------------------


def _build_manifest() -> dict:
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "stimuli": [
            {"id": s.stimulus_id, "description": s.description} for s in STIMULI
        ],
        "conditions": [c.name for c in CONDITIONS],
    }


def load_manifest() -> dict:
    return _build_manifest()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_suite(
    *,
    model_name: str,
    seed: int = 1337,
    limit: int | None = None,
    condition_names: list[str] | None = None,
    model: AudioProcessor | None = None,
) -> dict:
    manifest = _build_manifest()
    stimuli = list(STIMULI[:limit]) if limit else list(STIMULI)
    if not stimuli:
        raise ValueError("no stimuli selected")

    selected_conditions = list(CONDITIONS)
    if condition_names:
        allowed = set(condition_names)
        selected_conditions = [c for c in selected_conditions if c.name in allowed]
    if not selected_conditions:
        raise ValueError("no conditions selected")

    processor = model if model is not None else make_model(model_name)

    per_condition_si_sdr: dict[str, list[float]] = {c.name: [] for c in selected_conditions}
    per_condition_mr_stft: dict[str, list[float]] = {c.name: [] for c in selected_conditions}
    per_condition_true_peak: dict[str, list[float]] = {c.name: [] for c in selected_conditions}
    per_condition_loudness_delta: dict[str, list[float]] = {c.name: [] for c in selected_conditions}
    per_stimulus: list[dict] = []

    for stimulus in stimuli:
        ref_audio = stimulus.factory()
        results_for_stim: dict[str, dict] = {}
        for condition in selected_conditions:
            input_audio = condition.transform(ref_audio)
            output_audio, output_sr = processor.process(input_audio, SAMPLE_RATE)
            output_audio = np.asarray(output_audio, dtype=np.float32)
            si_sdr = si_sdr_db(input_audio, output_audio)
            mr_stft = mr_stft_log_l1(input_audio, output_audio)
            ref_tp = true_peak_dbtp(input_audio)
            out_tp = true_peak_dbtp(output_audio)
            ref_lufs = k_weighted_loudness_lufs(input_audio, SAMPLE_RATE)
            out_lufs = k_weighted_loudness_lufs(output_audio, SAMPLE_RATE)
            results_for_stim[condition.name] = {
                "si_sdr_db": si_sdr,
                "mr_stft_log_l1": mr_stft,
                "true_peak_input_dbtp": ref_tp,
                "true_peak_output_dbtp": out_tp,
                "true_peak_delta_db": out_tp - ref_tp,
                "loudness_input_lufs": ref_lufs,
                "loudness_output_lufs": out_lufs,
                "loudness_delta_lu": out_lufs - ref_lufs,
                "output_sample_rate": int(output_sr),
                "sample_rate_match": int(output_sr) == SAMPLE_RATE,
            }
            per_condition_si_sdr[condition.name].append(si_sdr)
            per_condition_mr_stft[condition.name].append(mr_stft)
            per_condition_true_peak[condition.name].append(out_tp)
            per_condition_loudness_delta[condition.name].append(out_lufs - ref_lufs)
        per_stimulus.append({
            "stimulus_id": stimulus.stimulus_id,
            "description": stimulus.description,
            "per_condition": results_for_stim,
        })

    def _avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    per_condition_metrics = {
        c.name: {
            "mean_si_sdr_db": _avg(per_condition_si_sdr[c.name]),
            "mean_mr_stft_log_l1": _avg(per_condition_mr_stft[c.name]),
            "mean_true_peak_dbtp": _avg(per_condition_true_peak[c.name]),
            "max_true_peak_dbtp": max(per_condition_true_peak[c.name]) if per_condition_true_peak[c.name] else -120.0,
            "mean_loudness_delta_lu": _avg(per_condition_loudness_delta[c.name]),
        }
        for c in selected_conditions
    }

    all_si_sdr = [v for vals in per_condition_si_sdr.values() for v in vals]
    all_true_peak = [v for vals in per_condition_true_peak.values() for v in vals]
    all_loudness_delta = [v for vals in per_condition_loudness_delta.values() for v in vals]

    headline = {
        "weighted_si_sdr_db": _avg(all_si_sdr),
        "max_true_peak_dbtp": max(all_true_peak) if all_true_peak else -120.0,
        "mean_loudness_delta_lu": _avg(all_loudness_delta),
        "stimulus_count": len(stimuli),
        "condition_count": len(selected_conditions),
    }

    config = {
        "model": model_name,
        "seed": seed,
        "stimulus_count": len(stimuli),
        "conditions": [c.name for c in selected_conditions],
        "sample_rate": SAMPLE_RATE,
    }
    digest = manifest_hash(manifest)
    digest_run = run_hash(
        suite=SUITE_ID,
        revision=SUITE_REVISION,
        manifest_digest=digest,
        config=config,
        hypotheses=per_stimulus,
    )
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "model": processor.name,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "stimulus_count": len(stimuli),
        "conditions": [c.name for c in selected_conditions],
        "manifest_hash": digest,
        "headline": headline,
        "per_condition_metrics": per_condition_metrics,
        "per_stimulus": per_stimulus,
        "run_hash": digest_run,
    }
