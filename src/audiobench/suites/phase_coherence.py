"""ab/phase-coherence — phase and multichannel coherence regressions.

Bundled stereo fixtures cover the most common phase mistakes a processor can
make:

- ``identity`` — should preserve inter-channel correlation
- ``polarity-pair`` — L = +tone, R = -tone; processor must keep them opposed
- ``ms-roundtrip`` — Mid/Side reconstruction must be lossless
- ``sub-sample-delay`` — small inter-channel delay (Δt < 1 sample) must survive

Headline:
- ``phase_coherence_score`` (mean per-stimulus pass rate, 1.0 best)
- ``mean_polarity_score`` (fraction of channels that kept their polarity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audiobench.hashing import manifest_hash, run_hash
from audiobench.models.audio_processor import AudioProcessor
from audiobench.models.signal_registry import make_model
from audiobench.signal_metrics import (
    interchannel_correlation,
    mid_side_round_trip_snr_db,
    polarity_preservation_score,
)


SUITE_ID = "ab/phase-coherence"
SUITE_REVISION = "0.1.0"
SAMPLE_RATE = 16000
POLARITY_PASS_MIN = 0.99
CORRELATION_TOLERANCE = 0.05
MS_ROUNDTRIP_MIN_SNR_DB = 40.0


def _tone(freq: float, duration_s: float, *, amplitude: float = 0.4, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * math.pi * freq * t + phase)).astype(np.float32)


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=0)


def _delay_subsample(audio: np.ndarray, delay_samples: float) -> np.ndarray:
    """Fractional-sample delay via spectral phase shift."""
    n = len(audio)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * (delay_samples / SAMPLE_RATE))
    delayed = np.fft.irfft(spectrum * phase_shift, n=n)
    return delayed.astype(np.float32)


@dataclass(frozen=True)
class PhaseStimulus:
    stimulus_id: str
    description: str
    check: str  # one of: "correlation", "polarity", "ms_roundtrip"
    expected_correlation: float | None
    audio: np.ndarray


def _build_stimuli() -> tuple[PhaseStimulus, ...]:
    out: list[PhaseStimulus] = []

    base = _tone(440.0, 1.0)
    out.append(PhaseStimulus(
        stimulus_id="identity-stereo",
        description="Mono content duplicated to L and R (corr=1)",
        check="correlation",
        expected_correlation=1.0,
        audio=_stereo(base, base.copy()),
    ))

    polarity_pair = _stereo(base, -base)
    out.append(PhaseStimulus(
        stimulus_id="polarity-pair",
        description="L = tone, R = -tone (corr=-1, polarity flip stress)",
        check="polarity",
        expected_correlation=-1.0,
        audio=polarity_pair,
    ))

    quad_phase = _stereo(
        _tone(440.0, 1.0, phase=0.0),
        _tone(440.0, 1.0, phase=math.pi / 2.0),
    )
    out.append(PhaseStimulus(
        stimulus_id="quad-phase-pair",
        description="L and R 90° apart (corr≈0)",
        check="correlation",
        expected_correlation=0.0,
        audio=quad_phase,
    ))

    mid = _tone(660.0, 1.0)
    side = _tone(330.0, 1.0, amplitude=0.2)
    ms_left = mid + side
    ms_right = mid - side
    out.append(PhaseStimulus(
        stimulus_id="ms-roundtrip",
        description="Mid/side encoded stereo — reconstruction must be lossless",
        check="ms_roundtrip",
        expected_correlation=None,
        audio=_stereo(ms_left, ms_right),
    ))

    delayed_right = _delay_subsample(base, 0.4)
    out.append(PhaseStimulus(
        stimulus_id="sub-sample-delay",
        description="R delayed by 0.4 samples (inter-aural time difference)",
        check="correlation",
        expected_correlation=0.99,
        audio=_stereo(base, delayed_right),
    ))

    return tuple(out)


_STIMULI = _build_stimuli()


def _build_manifest() -> dict:
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "stimuli": [
            {
                "id": s.stimulus_id,
                "description": s.description,
                "check": s.check,
                "expected_correlation": s.expected_correlation,
            }
            for s in _STIMULI
        ],
    }


def load_manifest() -> dict:
    return _build_manifest()


def _stimulus_passes(
    stimulus: PhaseStimulus,
    input_audio: np.ndarray,
    output_audio: np.ndarray,
) -> tuple[bool, dict]:
    metrics: dict = {}
    if stimulus.check == "correlation":
        corr_in = interchannel_correlation(input_audio)
        corr_out = interchannel_correlation(output_audio)
        metrics["correlation_input"] = corr_in
        metrics["correlation_output"] = corr_out
        metrics["correlation_delta"] = corr_out - corr_in
        if stimulus.expected_correlation is None:
            return abs(corr_out - corr_in) <= CORRELATION_TOLERANCE, metrics
        return abs(corr_out - stimulus.expected_correlation) <= CORRELATION_TOLERANCE * 4, metrics
    if stimulus.check == "polarity":
        score = polarity_preservation_score(input_audio, output_audio)
        corr_out = interchannel_correlation(output_audio)
        metrics["polarity_score"] = score
        metrics["correlation_output"] = corr_out
        return score >= POLARITY_PASS_MIN and corr_out < 0, metrics
    if stimulus.check == "ms_roundtrip":
        snr = mid_side_round_trip_snr_db(output_audio)
        metrics["ms_roundtrip_snr_db"] = snr
        return snr >= MS_ROUNDTRIP_MIN_SNR_DB, metrics
    return False, metrics


def run_suite(
    *,
    model_name: str,
    seed: int = 1337,
    limit: int | None = None,
    model: AudioProcessor | None = None,
) -> dict:
    manifest = _build_manifest()
    stimuli = list(_STIMULI[:limit]) if limit else list(_STIMULI)
    if not stimuli:
        raise ValueError("no stimuli selected")

    processor = model if model is not None else make_model(model_name)

    per_stimulus: list[dict] = []
    passed = 0
    polarity_scores: list[float] = []

    for stimulus in stimuli:
        input_audio = stimulus.audio
        output_audio, _ = processor.process(input_audio, SAMPLE_RATE)
        output_audio = np.asarray(output_audio, dtype=np.float32)
        if output_audio.ndim == 1:
            output_audio = np.stack([output_audio, output_audio], axis=0)
        ok, metrics = _stimulus_passes(stimulus, input_audio, output_audio)
        if ok:
            passed += 1
        polarity_scores.append(polarity_preservation_score(input_audio, output_audio))
        per_stimulus.append({
            "stimulus_id": stimulus.stimulus_id,
            "description": stimulus.description,
            "check": stimulus.check,
            "expected_correlation": stimulus.expected_correlation,
            "metrics": metrics,
            "passed": ok,
        })

    score = passed / float(len(stimuli))
    headline = {
        "phase_coherence_score": score,
        "passed_count": passed,
        "stimulus_count": len(stimuli),
        "mean_polarity_score": float(sum(polarity_scores) / len(polarity_scores)) if polarity_scores else 1.0,
    }

    config = {
        "model": model_name,
        "seed": seed,
        "stimulus_count": len(stimuli),
        "sample_rate": SAMPLE_RATE,
        "polarity_pass_min": POLARITY_PASS_MIN,
        "correlation_tolerance": CORRELATION_TOLERANCE,
        "ms_roundtrip_min_snr_db": MS_ROUNDTRIP_MIN_SNR_DB,
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
        "manifest_hash": digest,
        "headline": headline,
        "per_stimulus": per_stimulus,
        "run_hash": digest_run,
    }
