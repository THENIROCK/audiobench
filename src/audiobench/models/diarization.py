"""Speaker diarization adapter contract + reference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DiarizationAdapter(Protocol):
    """A diarization model returns a list of speaker turns for the input audio.

    Each turn is ``{"speaker_id": str, "start_s": float, "end_s": float}``.
    Speaker ids are anonymous — DER scoring aligns them to the reference.
    """

    name: str

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        ...


@dataclass
class OracleDiarizationAdapter:
    """Returns the ground-truth turns exactly — sanity-check upper bound."""

    name: str = "oracle-diarization"
    _hint: list[dict] | None = None

    def set_oracle_hint(self, turns: list[dict]) -> None:
        self._hint = [dict(turn) for turn in turns]

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        return list(self._hint or [])


@dataclass
class MergedDiarizationAdapter:
    """Collapses every speaker into one — heavy confusion error (regression demo)."""

    name: str = "merged-diarization"
    _hint: list[dict] | None = None

    def set_oracle_hint(self, turns: list[dict]) -> None:
        self._hint = [dict(turn) for turn in turns]

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        return [
            {"speaker_id": "spk-merged", "start_s": turn["start_s"], "end_s": turn["end_s"]}
            for turn in (self._hint or [])
        ]


@dataclass
class SingleSpeakerAdapter:
    """Emits one huge turn covering the whole clip — heavy false-alarm error."""

    name: str = "single-speaker"

    def set_oracle_hint(self, turns: list[dict]) -> None:  # noqa: D401 - intentional no-op
        return None

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        if audio.ndim == 1:
            duration = float(len(audio)) / float(sample_rate)
        else:
            duration = float(audio.shape[-1]) / float(sample_rate)
        return [{"speaker_id": "spk-only", "start_s": 0.0, "end_s": duration}]


def make_oracle_diarization() -> OracleDiarizationAdapter:
    return OracleDiarizationAdapter()


def make_merged_diarization() -> MergedDiarizationAdapter:
    return MergedDiarizationAdapter()


def make_single_speaker() -> SingleSpeakerAdapter:
    return SingleSpeakerAdapter()
