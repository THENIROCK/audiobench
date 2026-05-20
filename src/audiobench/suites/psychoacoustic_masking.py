"""ab/psychoacoustic-masking — does the processor respect audibility?

Each stimulus is a calibrated tone-in-noise pair where we know whether the
target tone is psychoacoustically audible or masked. After the processor
runs, we check:

- For *audible* stimuli, that the in-band SNR around the target tone stays
  positive (the tone was preserved).
- For *inaudible* (masked) stimuli, that the processor did not invent or
  amplify the target tone above its masking threshold.

Headline:
- ``masking_respect_score`` (1.0 = perfect; the processor kept audible tones
  audible and masked tones inaudible)
- ``mean_in_band_snr_delta_db`` for the audible stimuli (closer to 0 is better)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audiobench.hashing import manifest_hash, run_hash
from audiobench.models.audio_processor import AudioProcessor
from audiobench.models.signal_registry import make_model
from audiobench.signal_metrics import band_energy_db, band_snr_db


SUITE_ID = "ab/psychoacoustic-masking"
SUITE_REVISION = "0.1.0"
SAMPLE_RATE = 16000


def _tone(freq: float, duration_s: float, amplitude: float, *, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * math.pi * freq * t + phase)).astype(np.float32)


def _pink_noise(duration_s: float, *, seed: int, amplitude: float = 0.2) -> np.ndarray:
    """Approximate 1/f noise via Voss-McCartney."""
    n = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(seed)
    n_rows = 16
    array = np.zeros((n_rows, n), dtype=np.float64)
    for row in range(n_rows):
        step = 2 ** row
        positions = np.arange(0, n, step)
        array[row, positions] = rng.standard_normal(len(positions))
        for i in range(1, step):
            if i >= n:
                break
            array[row, i::step] = array[row, ::step][: len(array[row, i::step])]
    pink = array.sum(axis=0)
    pink /= max(np.std(pink), 1e-9)
    return (amplitude * pink).astype(np.float32)


def _narrowband_noise(center_hz: float, duration_s: float, *, seed: int, amplitude: float = 0.2) -> np.ndarray:
    from scipy import signal as scipy_signal

    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(int(SAMPLE_RATE * duration_s))
    bw = max(center_hz * 0.3, 100.0)
    low = max(50.0, center_hz - bw)
    high = min(SAMPLE_RATE * 0.49, center_hz + bw)
    sos = scipy_signal.butter(6, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, audio)
    filtered /= max(np.std(filtered), 1e-9)
    return (amplitude * filtered).astype(np.float32)


@dataclass(frozen=True)
class MaskingStimulus:
    stimulus_id: str
    description: str
    target_hz: float
    expected_audible: bool
    snr_band_hz: tuple[float, float]

    def render(self) -> np.ndarray: ...


@dataclass(frozen=True)
class _BuiltStimulus:
    stimulus_id: str
    description: str
    target_hz: float
    expected_audible: bool
    snr_band_hz: tuple[float, float]
    audio: np.ndarray


def _build_stimuli() -> tuple[_BuiltStimulus, ...]:
    out: list[_BuiltStimulus] = []

    # 1 kHz tone +6 dB above pink noise — clearly audible.
    tone = _tone(1000.0, 1.0, amplitude=0.35)
    noise = _pink_noise(1.0, seed=11, amplitude=0.2)
    out.append(_BuiltStimulus(
        stimulus_id="tone-1k-audible",
        description="1 kHz tone @ +6 dB over pink noise (audible)",
        target_hz=1000.0,
        expected_audible=True,
        snr_band_hz=(900.0, 1100.0),
        audio=(tone + noise).astype(np.float32),
    ))

    # 1 kHz tone deeply buried in noise — perceptually inaudible.
    weak_tone = _tone(1000.0, 1.0, amplitude=0.005)
    loud_noise = _pink_noise(1.0, seed=13, amplitude=0.3)
    out.append(_BuiltStimulus(
        stimulus_id="tone-1k-masked",
        description="1 kHz tone deeply masked by pink noise",
        target_hz=1000.0,
        expected_audible=False,
        snr_band_hz=(900.0, 1100.0),
        audio=(weak_tone + loud_noise).astype(np.float32),
    ))

    # 500 Hz tone with masker centered at 2 kHz (cross-band → audible).
    tone = _tone(500.0, 1.0, amplitude=0.3)
    masker = _narrowband_noise(2000.0, 1.0, seed=17, amplitude=0.25)
    out.append(_BuiltStimulus(
        stimulus_id="tone-500-cross-band-audible",
        description="500 Hz tone with 2 kHz narrow-band masker (different critical band)",
        target_hz=500.0,
        expected_audible=True,
        snr_band_hz=(400.0, 600.0),
        audio=(tone + masker).astype(np.float32),
    ))

    # 4 kHz quiet tone alone — should pass through without artifacts.
    out.append(_BuiltStimulus(
        stimulus_id="quiet-tone-4k",
        description="4 kHz tone at -40 dBFS, no masker",
        target_hz=4000.0,
        expected_audible=True,
        snr_band_hz=(3800.0, 4200.0),
        audio=_tone(4000.0, 1.0, amplitude=0.01),
    ))

    # Pure pink noise, no target — processor must not invent a 1 kHz tone.
    out.append(_BuiltStimulus(
        stimulus_id="no-tone-pink-only",
        description="Pink noise alone — should remain tone-free",
        target_hz=1000.0,
        expected_audible=False,
        snr_band_hz=(950.0, 1050.0),
        audio=_pink_noise(1.0, seed=19, amplitude=0.25),
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
                "target_hz": s.target_hz,
                "expected_audible": s.expected_audible,
                "snr_band_hz": list(s.snr_band_hz),
            }
            for s in _STIMULI
        ],
    }


def load_manifest() -> dict:
    return _build_manifest()


# A stimulus is "respected" if the processor's behavior aligns with its
# expected audibility:
#   audible    -> output keeps in-band SNR above AUDIBLE_SNR_MIN_DB
#   inaudible  -> output keeps in-band energy near (or below) the input's
AUDIBLE_SNR_MIN_DB = -3.0
INAUDIBLE_BAND_DELTA_MAX_DB = 3.0


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
    respected = 0
    audible_snr_deltas: list[float] = []
    inaudible_energy_deltas: list[float] = []

    for stimulus in stimuli:
        input_audio = stimulus.audio.astype(np.float32)
        output_audio, _ = processor.process(input_audio, SAMPLE_RATE)
        output_audio = np.asarray(output_audio, dtype=np.float32)
        low, high = stimulus.snr_band_hz
        snr_in = band_snr_db(input_audio, input_audio, SAMPLE_RATE, low, high)
        snr_out = band_snr_db(input_audio, output_audio, SAMPLE_RATE, low, high)
        energy_in = band_energy_db(input_audio, SAMPLE_RATE, low, high)
        energy_out = band_energy_db(output_audio, SAMPLE_RATE, low, high)
        energy_delta = energy_out - energy_in
        snr_delta = snr_out - snr_in

        if stimulus.expected_audible:
            respected_here = snr_out >= AUDIBLE_SNR_MIN_DB
            audible_snr_deltas.append(snr_delta)
        else:
            respected_here = abs(energy_delta) <= INAUDIBLE_BAND_DELTA_MAX_DB
            inaudible_energy_deltas.append(energy_delta)

        if respected_here:
            respected += 1

        per_stimulus.append({
            "stimulus_id": stimulus.stimulus_id,
            "description": stimulus.description,
            "target_hz": stimulus.target_hz,
            "expected_audible": stimulus.expected_audible,
            "snr_band_hz": list(stimulus.snr_band_hz),
            "input_band_energy_db": energy_in,
            "output_band_energy_db": energy_out,
            "input_band_snr_db": snr_in,
            "output_band_snr_db": snr_out,
            "in_band_snr_delta_db": snr_delta,
            "in_band_energy_delta_db": energy_delta,
            "respected": respected_here,
        })

    def _avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    headline = {
        "masking_respect_score": respected / float(len(stimuli)),
        "respected_count": respected,
        "stimulus_count": len(stimuli),
        "mean_in_band_snr_delta_db": _avg(audible_snr_deltas),
        "mean_inaudible_energy_delta_db": _avg(inaudible_energy_deltas),
    }

    config = {
        "model": model_name,
        "seed": seed,
        "stimulus_count": len(stimuli),
        "sample_rate": SAMPLE_RATE,
        "audible_snr_min_db": AUDIBLE_SNR_MIN_DB,
        "inaudible_band_delta_max_db": INAUDIBLE_BAND_DELTA_MAX_DB,
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
