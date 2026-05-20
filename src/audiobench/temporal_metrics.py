"""Temporal evaluation metrics shared by Phase 4 task suites.

Covers two families:

- **Event detection** (ab/sed-urban): per-label IoU-matched event F1, plus
  1-second segment F1 (presence/absence per label per segment).
- **Speaker diarization** (ab/diarization-cw): Diarization Error Rate (DER)
  with a configurable collar, using Hungarian alignment of hypothesis
  speakers to reference speakers.

Each metric works on pure Python dicts / dataclass-like dicts; no audio is
read in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Event detection metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    label: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {"label": self.label, "start_s": self.start_s, "end_s": self.end_s}


def _events_from_payload(items: Iterable[dict | Event]) -> list[Event]:
    out: list[Event] = []
    for item in items:
        if isinstance(item, Event):
            out.append(item)
            continue
        out.append(
            Event(
                label=str(item["label"]),
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
            )
        )
    return out


def event_iou(a: Event, b: Event) -> float:
    if a.label != b.label:
        return 0.0
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    if inter == 0.0:
        return 0.0
    union = a.duration + b.duration - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def event_f1(
    reference: Sequence[dict | Event],
    hypothesis: Sequence[dict | Event],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Greedy per-label IoU match → micro precision/recall/F1."""
    refs = _events_from_payload(reference)
    hyps = _events_from_payload(hypothesis)
    matched_ref: set[int] = set()
    matched_hyp: set[int] = set()
    # Greedy: highest IoU first.
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(refs):
        for j, h in enumerate(hyps):
            iou = event_iou(r, h)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda t: t[0], reverse=True)
    for iou, i, j in candidates:
        if i in matched_ref or j in matched_hyp:
            continue
        matched_ref.add(i)
        matched_hyp.add(j)
    tp = len(matched_ref)
    fn = len(refs) - tp
    fp = len(hyps) - tp
    if tp + fp + fn == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou_threshold": iou_threshold,
        }
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": iou_threshold,
    }


def _segment_grid(
    events: Sequence[Event],
    duration_s: float,
    segment_s: float,
    labels: Sequence[str],
) -> np.ndarray:
    n_segments = max(1, int(math.ceil(duration_s / segment_s)))
    grid = np.zeros((len(labels), n_segments), dtype=np.int8)
    label_index = {label: idx for idx, label in enumerate(labels)}
    for ev in events:
        if ev.label not in label_index:
            continue
        start = max(0.0, ev.start_s)
        end = min(duration_s, ev.end_s)
        if end <= start:
            continue
        first = int(math.floor(start / segment_s))
        last = int(math.ceil(end / segment_s))
        grid[label_index[ev.label], first:last] = 1
    return grid


def segment_f1(
    reference: Sequence[dict | Event],
    hypothesis: Sequence[dict | Event],
    *,
    duration_s: float,
    segment_s: float = 1.0,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """1-second segment F1 (sklearn-style micro), label-aware."""
    refs = _events_from_payload(reference)
    hyps = _events_from_payload(hypothesis)
    if labels is None:
        labels = sorted({e.label for e in refs} | {e.label for e in hyps})
    if not labels:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "segment_s": segment_s,
            "segment_count": 0,
        }
    ref_grid = _segment_grid(refs, duration_s, segment_s, labels)
    hyp_grid = _segment_grid(hyps, duration_s, segment_s, labels)
    tp = int(np.sum((ref_grid == 1) & (hyp_grid == 1)))
    fp = int(np.sum((ref_grid == 0) & (hyp_grid == 1)))
    fn = int(np.sum((ref_grid == 1) & (hyp_grid == 0)))
    if tp + fp + fn == 0:
        # No reference events and no hypothesis events anywhere — perfect score.
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "segment_s": segment_s,
            "segment_count": int(ref_grid.shape[1]),
        }
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "segment_s": segment_s,
        "segment_count": int(ref_grid.shape[1]),
    }


# ---------------------------------------------------------------------------
# Diarization metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    speaker_id: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {"speaker_id": self.speaker_id, "start_s": self.start_s, "end_s": self.end_s}


def _turns_from_payload(items: Iterable[dict | Turn]) -> list[Turn]:
    out: list[Turn] = []
    for item in items:
        if isinstance(item, Turn):
            out.append(item)
            continue
        out.append(
            Turn(
                speaker_id=str(item["speaker_id"]),
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
            )
        )
    return out


def _frame_labels(
    turns: Sequence[Turn],
    duration_s: float,
    frame_s: float,
) -> list[set[str]]:
    n = max(1, int(math.ceil(duration_s / frame_s)))
    labels: list[set[str]] = [set() for _ in range(n)]
    for turn in turns:
        start = max(0.0, turn.start_s)
        end = min(duration_s, turn.end_s)
        if end <= start:
            continue
        first = int(math.floor(start / frame_s))
        last = int(math.ceil(end / frame_s))
        for i in range(first, min(n, last)):
            labels[i].add(turn.speaker_id)
    return labels


def _frame_collar_mask(
    turns: Sequence[Turn],
    duration_s: float,
    frame_s: float,
    collar_s: float,
) -> np.ndarray:
    """1 where the frame is OUTSIDE any reference collar (eligible for scoring)."""
    n = max(1, int(math.ceil(duration_s / frame_s)))
    if collar_s <= 0:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for turn in turns:
        boundaries = (turn.start_s, turn.end_s)
        for boundary in boundaries:
            lo = max(0.0, boundary - collar_s)
            hi = min(duration_s, boundary + collar_s)
            first = int(math.floor(lo / frame_s))
            last = int(math.ceil(hi / frame_s))
            mask[first:last] = False
    return mask


def _hungarian_min(cost: np.ndarray) -> list[int]:
    """Simple Hungarian assignment for small square cost matrices.

    Falls back to greedy nearest match when the matrix is large enough that
    the brute-force exact search would be expensive.
    """
    n, m = cost.shape
    size = max(n, m)
    if size == 0:
        return []
    padded = np.full((size, size), cost.max() + 1.0, dtype=np.float64)
    padded[:n, :m] = cost
    if size <= 6:
        from itertools import permutations

        best_perm = None
        best_total = float("inf")
        for perm in permutations(range(size)):
            total = sum(padded[r, c] for r, c in enumerate(perm))
            if total < best_total:
                best_total = total
                best_perm = perm
        return list(best_perm or range(size))
    # Greedy fallback.
    assigned_cols: set[int] = set()
    out: list[int] = [0] * size
    for r in range(size):
        order = np.argsort(padded[r])
        for c in order:
            c_int = int(c)
            if c_int not in assigned_cols:
                out[r] = c_int
                assigned_cols.add(c_int)
                break
    return out


def diarization_error_rate(
    reference: Sequence[dict | Turn],
    hypothesis: Sequence[dict | Turn],
    *,
    duration_s: float,
    frame_s: float = 0.05,
    collar_s: float = 0.25,
) -> dict[str, float | int]:
    """Compute DER = (FA + Miss + Confusion) / total reference speech.

    Implements the canonical NIST recipe at frame granularity with a symmetric
    collar around every reference turn boundary. Overlap is supported: a frame
    can carry multiple reference speakers.
    """
    refs = _turns_from_payload(reference)
    hyps = _turns_from_payload(hypothesis)
    ref_frames = _frame_labels(refs, duration_s, frame_s)
    hyp_frames = _frame_labels(hyps, duration_s, frame_s)
    mask = _frame_collar_mask(refs, duration_s, frame_s, collar_s)

    ref_speakers = sorted({t.speaker_id for t in refs})
    hyp_speakers = sorted({t.speaker_id for t in hyps})
    if not ref_speakers:
        total = sum(1 for frame in hyp_frames if frame)
        return {
            "der": 0.0 if total == 0 else 1.0,
            "false_alarm_rate": float(total),
            "miss_rate": 0.0,
            "confusion_rate": 0.0,
            "speech_frames": 0,
            "speaker_count_reference": 0,
            "speaker_count_hypothesis": len(hyp_speakers),
            "speaker_count_error": len(hyp_speakers),
        }

    cost = np.zeros((len(ref_speakers), len(hyp_speakers) or 1), dtype=np.float64)
    if hyp_speakers:
        ref_idx = {sp: i for i, sp in enumerate(ref_speakers)}
        hyp_idx = {sp: i for i, sp in enumerate(hyp_speakers)}
        for idx, (rset, hset, eligible) in enumerate(zip(ref_frames, hyp_frames, mask)):
            if not eligible:
                continue
            for rsp in rset:
                for hsp in hset:
                    cost[ref_idx[rsp], hyp_idx[hsp]] -= 1.0
        assignment = _hungarian_min(cost)
        # Map ref speaker -> hyp speaker (when both exist).
        ref_to_hyp: dict[str, str] = {}
        for r, c in enumerate(assignment):
            if r < len(ref_speakers) and c < len(hyp_speakers):
                ref_to_hyp[ref_speakers[r]] = hyp_speakers[c]
    else:
        ref_to_hyp = {}

    miss = 0
    false_alarm = 0
    confusion = 0
    speech_frames = 0
    for rset, hset, eligible in zip(ref_frames, hyp_frames, mask):
        if not eligible:
            continue
        n_ref = len(rset)
        n_hyp = len(hset)
        speech_frames += n_ref
        # Frames with no reference speech contribute pure false alarm.
        if n_ref == 0:
            false_alarm += n_hyp
            continue
        # Count correctly aligned pairs in this frame.
        aligned_hits = 0
        for rsp in rset:
            mapped = ref_to_hyp.get(rsp)
            if mapped is not None and mapped in hset:
                aligned_hits += 1
        # Standard NIST decomposition.
        miss += max(0, n_ref - n_hyp)
        false_alarm += max(0, n_hyp - n_ref)
        confusion += min(n_ref, n_hyp) - aligned_hits

    total_speech = max(1, speech_frames)
    miss_rate = miss / total_speech
    fa_rate = false_alarm / total_speech
    confusion_rate = confusion / total_speech
    der = miss_rate + fa_rate + confusion_rate

    return {
        "der": float(der),
        "false_alarm_rate": float(fa_rate),
        "miss_rate": float(miss_rate),
        "confusion_rate": float(confusion_rate),
        "speech_frames": int(speech_frames),
        "speaker_count_reference": len(ref_speakers),
        "speaker_count_hypothesis": len(hyp_speakers),
        "speaker_count_error": abs(len(ref_speakers) - len(hyp_speakers)),
    }
