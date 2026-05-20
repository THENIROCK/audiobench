"""Sound event detection (SED) adapter contract + reference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SEDAdapter(Protocol):
    """A SED model returns a list of events for the input audio.

    Each event is a dict ``{"label": str, "start_s": float, "end_s": float}``.
    Optionally include ``"confidence": float`` (ignored by the bundled scorer).
    """

    name: str

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        ...


# The procedural SED suite passes the ground truth through ``hints`` to the
# adapter via ``set_oracle_hint`` so the oracle baselines can answer perfectly.
# Real adapters ignore that field and rely purely on the audio.


@dataclass
class OracleSEDAdapter:
    """Returns the ground-truth events exactly — sanity-check upper bound."""

    name: str = "oracle-sed"
    _hint: list[dict] | None = None

    def set_oracle_hint(self, events: list[dict]) -> None:
        self._hint = [dict(ev) for ev in events]

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        return list(self._hint or [])


@dataclass
class JitteredOracleSEDAdapter:
    """Oracle, but every event boundary is shifted by ``jitter_s`` (regression demo)."""

    name: str = "oracle-sed-jittered"
    jitter_s: float = 0.4
    _hint: list[dict] | None = None
    _seq: int = 0

    def set_oracle_hint(self, events: list[dict]) -> None:
        self._hint = [dict(ev) for ev in events]

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        out = []
        for idx, ev in enumerate(self._hint or []):
            sign = 1.0 if (idx + self._seq) % 2 == 0 else -1.0
            out.append({
                "label": ev["label"],
                "start_s": max(0.0, ev["start_s"] + sign * self.jitter_s),
                "end_s": max(0.0, ev["end_s"] + sign * self.jitter_s),
            })
        self._seq += 1
        return out


@dataclass
class NullSEDAdapter:
    """Returns nothing — worst case for recall."""

    name: str = "null-sed"

    def set_oracle_hint(self, events: list[dict]) -> None:  # noqa: D401 - intentional no-op
        return None

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        return []


def make_oracle_sed() -> OracleSEDAdapter:
    return OracleSEDAdapter()


def make_jittered_oracle_sed() -> JitteredOracleSEDAdapter:
    return JitteredOracleSEDAdapter()


def make_null_sed() -> NullSEDAdapter:
    return NullSEDAdapter()
