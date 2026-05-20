"""ab/diarization-cw — speaker diarization (Conversational Workloads).

Each clip is a procedural multi-speaker conversation. Two or three "speakers"
are rendered as distinct formant-like signals and alternated according to a
fixed turn schedule. Adapters return ``[{"speaker_id", "start_s", "end_s"}]``;
we score Diarization Error Rate (DER) at 50 ms frame granularity with a
0.25 s collar, after Hungarian-aligning hypothesis speakers to references.

Headline:
- ``der`` (lower is better; standard NIST DER)
- ``speaker_count_error`` (lower is better)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audiobench.hashing import manifest_hash, run_hash
from audiobench.models.diarization import DiarizationAdapter
from audiobench.models.diarization_registry import make_model
from audiobench.temporal_metrics import diarization_error_rate


SUITE_ID = "ab/diarization-cw"
SUITE_REVISION = "0.1.0"
SAMPLE_RATE = 16000
FRAME_S = 0.05
COLLAR_S = 0.25


@dataclass(frozen=True)
class DiarClip:
    clip_id: str
    duration_s: float
    turns: tuple[dict, ...]


CLIPS: tuple[DiarClip, ...] = (
    DiarClip(
        clip_id="cw-001",
        duration_s=10.0,
        turns=(
            {"speaker_id": "spk-A", "start_s": 0.5, "end_s": 3.0},
            {"speaker_id": "spk-B", "start_s": 3.2, "end_s": 5.5},
            {"speaker_id": "spk-A", "start_s": 6.0, "end_s": 8.0},
            {"speaker_id": "spk-B", "start_s": 8.2, "end_s": 9.5},
        ),
    ),
    DiarClip(
        clip_id="cw-002",
        duration_s=12.0,
        turns=(
            {"speaker_id": "spk-A", "start_s": 0.0, "end_s": 2.5},
            {"speaker_id": "spk-B", "start_s": 2.8, "end_s": 5.0},
            {"speaker_id": "spk-C", "start_s": 5.5, "end_s": 8.0},
            {"speaker_id": "spk-A", "start_s": 8.5, "end_s": 11.5},
        ),
    ),
    DiarClip(
        clip_id="cw-003",
        duration_s=8.0,
        turns=(
            {"speaker_id": "spk-A", "start_s": 0.5, "end_s": 4.5},
            {"speaker_id": "spk-B", "start_s": 4.6, "end_s": 7.5},
        ),
    ),
    DiarClip(
        clip_id="cw-004",  # one-speaker baseline
        duration_s=6.0,
        turns=(
            {"speaker_id": "spk-A", "start_s": 0.5, "end_s": 5.5},
        ),
    ),
    DiarClip(
        clip_id="cw-005",  # short overlap section
        duration_s=10.0,
        turns=(
            {"speaker_id": "spk-A", "start_s": 0.5, "end_s": 4.0},
            {"speaker_id": "spk-B", "start_s": 3.5, "end_s": 7.0},
            {"speaker_id": "spk-A", "start_s": 7.2, "end_s": 9.5},
        ),
    ),
)


def _speaker_signal(speaker_id: str, n_samples: int, *, seed: int) -> np.ndarray:
    """Render a deterministic, perceptually-distinct timbre per speaker."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SAMPLE_RATE
    # Two formants per "speaker", spaced to be unambiguous.
    formants = {
        "spk-A": (320.0, 1200.0),
        "spk-B": (220.0, 850.0),
        "spk-C": (420.0, 1500.0),
    }.get(speaker_id, (260.0, 1050.0))
    f1, f2 = formants
    excitation = 0.6 * np.sin(2.0 * math.pi * f1 * t) + 0.4 * np.sin(2.0 * math.pi * f2 * t)
    excitation += 0.05 * rng.standard_normal(n_samples)
    env = 0.5 * (1.0 + np.tanh(8.0 * (t - 0.05))) * (1.0 - np.exp(-3.0 * t))
    return (0.5 * env * excitation).astype(np.float32)


def render_clip(clip: DiarClip, *, seed: int) -> np.ndarray:
    n = int(SAMPLE_RATE * clip.duration_s)
    audio = np.zeros(n, dtype=np.float32)
    audio += 0.005 * np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    for idx, turn in enumerate(clip.turns):
        start = int(SAMPLE_RATE * float(turn["start_s"]))
        end = int(SAMPLE_RATE * float(turn["end_s"]))
        end = min(end, n)
        if end <= start:
            continue
        signal = _speaker_signal(
            turn["speaker_id"],
            end - start,
            seed=seed + idx,
        )
        audio[start:end] += signal
    np.clip(audio, -0.99, 0.99, out=audio)
    return audio


def _build_manifest() -> dict:
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "frame_s": FRAME_S,
        "collar_s": COLLAR_S,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "duration_s": clip.duration_s,
                "turns": [dict(turn) for turn in clip.turns],
            }
            for clip in CLIPS
        ],
    }


def load_manifest() -> dict:
    return _build_manifest()


def run_suite(
    *,
    model_name: str,
    seed: int = 1337,
    limit: int | None = None,
    frame_s: float | None = None,
    collar_s: float | None = None,
    model: DiarizationAdapter | None = None,
) -> dict:
    frame_s = FRAME_S if frame_s is None else float(frame_s)
    collar_s = COLLAR_S if collar_s is None else float(collar_s)
    manifest = _build_manifest()
    clips = list(CLIPS[:limit]) if limit else list(CLIPS)
    if not clips:
        raise ValueError("no clips selected")

    diarizer = model if model is not None else make_model(model_name)

    per_clip: list[dict] = []
    total_speech_frames = 0
    weighted_miss = 0.0
    weighted_fa = 0.0
    weighted_conf = 0.0
    speaker_errors: list[int] = []

    for clip in clips:
        audio = render_clip(clip, seed=seed)
        if hasattr(diarizer, "set_oracle_hint"):
            diarizer.set_oracle_hint([dict(turn) for turn in clip.turns])
        hypothesis = diarizer.diarize(audio, SAMPLE_RATE) or []
        metrics = diarization_error_rate(
            list(clip.turns),
            hypothesis,
            duration_s=clip.duration_s,
            frame_s=frame_s,
            collar_s=collar_s,
        )
        per_clip.append({
            "clip_id": clip.clip_id,
            "duration_s": clip.duration_s,
            "reference_turns": [dict(turn) for turn in clip.turns],
            "hypothesis_turns": [dict(turn) for turn in hypothesis],
            "metrics": metrics,
        })
        speech_frames = int(metrics["speech_frames"])
        total_speech_frames += speech_frames
        weighted_miss += float(metrics["miss_rate"]) * speech_frames
        weighted_fa += float(metrics["false_alarm_rate"]) * speech_frames
        weighted_conf += float(metrics["confusion_rate"]) * speech_frames
        speaker_errors.append(int(metrics["speaker_count_error"]))

    if total_speech_frames > 0:
        miss = weighted_miss / total_speech_frames
        fa = weighted_fa / total_speech_frames
        conf = weighted_conf / total_speech_frames
    else:
        miss = fa = conf = 0.0
    der = miss + fa + conf

    headline = {
        "der": float(der),
        "miss_rate": float(miss),
        "false_alarm_rate": float(fa),
        "confusion_rate": float(conf),
        "mean_speaker_count_error": float(sum(speaker_errors) / max(1, len(speaker_errors))),
        "clip_count": len(clips),
        "frame_s": frame_s,
        "collar_s": collar_s,
    }

    config = {
        "model": model_name,
        "seed": seed,
        "clip_count": len(clips),
        "sample_rate": SAMPLE_RATE,
        "frame_s": frame_s,
        "collar_s": collar_s,
    }
    digest = manifest_hash(manifest)
    digest_run = run_hash(
        suite=SUITE_ID,
        revision=SUITE_REVISION,
        manifest_digest=digest,
        config=config,
        hypotheses=per_clip,
    )
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "model": diarizer.name,
        "seed": seed,
        "clip_count": len(clips),
        "sample_rate": SAMPLE_RATE,
        "manifest_hash": digest,
        "headline": headline,
        "per_clip": per_clip,
        "run_hash": digest_run,
    }
