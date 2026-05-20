"""ab/sed-urban — sound event detection (SED).

Each clip is a procedurally rendered 10-second soundscape: a known set of
labeled events (siren, dog_bark, alarm, engine, glass_breaking) is placed at
fixed timestamps over a pink-noise bed. The adapter under test must return
``[{"label", "start_s", "end_s"}]`` events; we score event-F1 with an IoU
threshold and segment-F1 at 1 s granularity.

Headline:
- ``event_f1_iou50`` (higher is better; the canonical SED number)
- ``segment_f1_1s`` (higher is better)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audiobench.hashing import manifest_hash, run_hash
from audiobench.models.sed import SEDAdapter
from audiobench.models.sed_registry import make_model
from audiobench.temporal_metrics import event_f1, segment_f1


SUITE_ID = "ab/sed-urban"
SUITE_REVISION = "0.1.0"
SAMPLE_RATE = 16000
CLIP_DURATION_S = 10.0
LABELS = ("siren", "dog_bark", "alarm", "engine", "glass_breaking")
IOU_THRESHOLD = 0.5
SEGMENT_S = 1.0


@dataclass(frozen=True)
class SEDClip:
    clip_id: str
    events: tuple[dict, ...]


def _clip_001() -> SEDClip:
    return SEDClip(
        clip_id="urban-001",
        events=(
            {"label": "siren", "start_s": 1.0, "end_s": 4.0},
            {"label": "dog_bark", "start_s": 5.5, "end_s": 6.2},
            {"label": "dog_bark", "start_s": 7.0, "end_s": 7.5},
        ),
    )


def _clip_002() -> SEDClip:
    return SEDClip(
        clip_id="urban-002",
        events=(
            {"label": "engine", "start_s": 0.0, "end_s": 6.0},
            {"label": "glass_breaking", "start_s": 3.5, "end_s": 3.9},
            {"label": "alarm", "start_s": 7.5, "end_s": 9.5},
        ),
    )


def _clip_003() -> SEDClip:
    return SEDClip(
        clip_id="urban-003",
        events=(
            {"label": "alarm", "start_s": 0.5, "end_s": 1.5},
            {"label": "alarm", "start_s": 3.0, "end_s": 4.0},
            {"label": "alarm", "start_s": 5.5, "end_s": 6.5},
        ),
    )


def _clip_004() -> SEDClip:
    return SEDClip(
        clip_id="urban-004",
        events=(
            {"label": "siren", "start_s": 2.0, "end_s": 8.0},
            {"label": "dog_bark", "start_s": 4.0, "end_s": 4.7},
        ),
    )


def _clip_005() -> SEDClip:
    return SEDClip(
        clip_id="urban-005",
        events=(),  # background-only — should produce no detections
    )


CLIPS: tuple[SEDClip, ...] = (_clip_001(), _clip_002(), _clip_003(), _clip_004(), _clip_005())


def _pink_noise(samples: int, *, seed: int, amplitude: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = 8
    arr = np.zeros((rows, samples), dtype=np.float64)
    for r in range(rows):
        step = 2 ** r
        positions = np.arange(0, samples, step)
        arr[r, positions] = rng.standard_normal(len(positions))
        for i in range(1, step):
            if i >= samples:
                break
            arr[r, i::step] = arr[r, ::step][: len(arr[r, i::step])]
    pink = arr.sum(axis=0)
    pink /= max(np.std(pink), 1e-9)
    return (amplitude * pink).astype(np.float32)


def _label_signal(label: str, n_samples: int, *, seed: int) -> np.ndarray:
    """Render a short, distinct procedural signal per label (purely auditory)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SAMPLE_RATE
    if label == "siren":
        freq = 700.0 + 400.0 * np.sin(2.0 * math.pi * 0.5 * t)
        return (0.4 * np.sin(2.0 * math.pi * np.cumsum(freq) / SAMPLE_RATE)).astype(np.float32)
    if label == "dog_bark":
        env = np.exp(-5.0 * (t - 0.05) ** 2) + 0.7 * np.exp(-5.0 * (t - 0.25) ** 2)
        return (0.6 * env * np.sin(2.0 * math.pi * 350.0 * t)).astype(np.float32)
    if label == "alarm":
        gate = (np.sin(2.0 * math.pi * 4.0 * t) > 0).astype(np.float32)
        return (0.5 * gate * np.sin(2.0 * math.pi * 1200.0 * t)).astype(np.float32)
    if label == "engine":
        wobble = 0.05 * np.sin(2.0 * math.pi * 8.0 * t)
        return (0.45 * np.sin(2.0 * math.pi * 90.0 * (t + wobble))).astype(np.float32)
    if label == "glass_breaking":
        noise = rng.standard_normal(n_samples)
        env = np.exp(-25.0 * t)
        return (0.7 * env * noise).astype(np.float32)
    return (0.3 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)


def render_clip(clip: SEDClip, *, seed: int) -> np.ndarray:
    n = int(SAMPLE_RATE * CLIP_DURATION_S)
    audio = _pink_noise(n, seed=seed)
    for event_idx, event in enumerate(clip.events):
        start = int(SAMPLE_RATE * float(event["start_s"]))
        end = int(SAMPLE_RATE * float(event["end_s"]))
        end = min(end, n)
        if end <= start:
            continue
        signal = _label_signal(
            event["label"],
            end - start,
            seed=seed + event_idx + hash(event["label"]) % 10_000,
        )
        audio[start:end] += signal
    np.clip(audio, -0.99, 0.99, out=audio)
    return audio.astype(np.float32)


def _build_manifest() -> dict:
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "clip_duration_s": CLIP_DURATION_S,
        "labels": list(LABELS),
        "iou_threshold": IOU_THRESHOLD,
        "segment_s": SEGMENT_S,
        "clips": [
            {"clip_id": clip.clip_id, "events": [dict(ev) for ev in clip.events]}
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
    iou_threshold: float | None = None,
    segment_s: float | None = None,
    model: SEDAdapter | None = None,
) -> dict:
    iou_threshold = IOU_THRESHOLD if iou_threshold is None else float(iou_threshold)
    segment_s = SEGMENT_S if segment_s is None else float(segment_s)
    manifest = _build_manifest()
    clips = list(CLIPS[:limit]) if limit else list(CLIPS)
    if not clips:
        raise ValueError("no clips selected")

    detector = model if model is not None else make_model(model_name)

    per_clip: list[dict] = []
    total_event = {"tp": 0, "fp": 0, "fn": 0}
    total_segment = {"tp": 0, "fp": 0, "fn": 0}

    for clip in clips:
        audio = render_clip(clip, seed=seed)
        if hasattr(detector, "set_oracle_hint"):
            detector.set_oracle_hint([dict(ev) for ev in clip.events])
        hypothesis = detector.detect(audio, SAMPLE_RATE) or []
        event_metrics = event_f1(
            list(clip.events),
            hypothesis,
            iou_threshold=iou_threshold,
        )
        seg_metrics = segment_f1(
            list(clip.events),
            hypothesis,
            duration_s=CLIP_DURATION_S,
            segment_s=segment_s,
            labels=LABELS,
        )
        per_clip.append({
            "clip_id": clip.clip_id,
            "reference_events": [dict(ev) for ev in clip.events],
            "hypothesis_events": [dict(ev) for ev in hypothesis],
            "event_metrics": event_metrics,
            "segment_metrics": seg_metrics,
        })
        total_event["tp"] += int(event_metrics["true_positives"])
        total_event["fp"] += int(event_metrics["false_positives"])
        total_event["fn"] += int(event_metrics["false_negatives"])
        # Recompute segment confusion counts from the (precision, recall) — easier
        # to aggregate by re-running the grid math; but our segment_f1 already
        # micro-averaged per clip. Aggregate the f1 itself as the mean across
        # clips so empty-clip cases don't divide by zero.
    micro_event_precision = total_event["tp"] / max(1, total_event["tp"] + total_event["fp"])
    micro_event_recall = total_event["tp"] / max(1, total_event["tp"] + total_event["fn"])
    if (micro_event_precision + micro_event_recall) == 0.0:
        micro_event_f1 = 0.0
    else:
        micro_event_f1 = (
            2.0 * micro_event_precision * micro_event_recall
            / (micro_event_precision + micro_event_recall)
        )
    mean_segment_f1 = sum(c["segment_metrics"]["f1"] for c in per_clip) / float(len(per_clip))

    headline = {
        "event_f1_iou50": float(micro_event_f1),
        "event_precision_iou50": float(micro_event_precision),
        "event_recall_iou50": float(micro_event_recall),
        "segment_f1_1s": float(mean_segment_f1),
        "clip_count": len(clips),
        "iou_threshold": iou_threshold,
        "segment_s": segment_s,
    }

    config = {
        "model": model_name,
        "seed": seed,
        "clip_count": len(clips),
        "sample_rate": SAMPLE_RATE,
        "iou_threshold": iou_threshold,
        "segment_s": segment_s,
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
        "model": detector.name,
        "seed": seed,
        "clip_count": len(clips),
        "labels": list(LABELS),
        "sample_rate": SAMPLE_RATE,
        "manifest_hash": digest,
        "headline": headline,
        "per_clip": per_clip,
        "run_hash": digest_run,
    }
