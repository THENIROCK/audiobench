"""Adapter protocol and response helpers for ASR suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeAlias

import numpy as np


@dataclass(frozen=True)
class ASRResult:
    """Normalized ASR response payload for per-sample reporting."""

    transcript: str
    latency_ms: float | None = None
    cost_usd: float | None = None
    error: str | None = None


ASRResponse: TypeAlias = str | ASRResult | Mapping[str, Any]


class ASRAdapter(Protocol):
    """Protocol for transcription models used by ASR suites."""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResponse: ...


def _coerce_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"invalid {field} value: {value!r}") from exc


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_asr_response(response: ASRResponse) -> ASRResult:
    """Normalize legacy/new adapter outputs into one JSON-safe shape.

    Existing adapters that return a plain transcript string continue to work.
    New adapters can return a mapping with rich metadata or ``ASRResult``.
    """

    if isinstance(response, ASRResult):
        return response
    if isinstance(response, str):
        return ASRResult(transcript=response.strip())
    if isinstance(response, Mapping):
        transcript = _coerce_text(response.get("transcript", response.get("text", "")))
        latency_ms = _coerce_float(response.get("latency_ms"), field="latency_ms")
        cost_usd = _coerce_float(response.get("cost_usd", response.get("cost")), field="cost_usd")
        error = response.get("error")
        if error is not None:
            error = str(error)
        return ASRResult(
            transcript=transcript,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            error=error,
        )
    raise TypeError(
        "ASR adapter transcribe() must return str, ASRResult, or mapping "
        f"(got {type(response).__name__})"
    )


def asr_result_to_dict(result: ASRResult) -> dict[str, Any]:
    """Return a compact, JSON-serializable record for run artifacts."""

    data: dict[str, Any] = {"transcript": result.transcript}
    if result.latency_ms is not None:
        data["latency_ms"] = result.latency_ms
    if result.cost_usd is not None:
        data["cost_usd"] = result.cost_usd
    if result.error is not None:
        data["error"] = result.error
    return data
