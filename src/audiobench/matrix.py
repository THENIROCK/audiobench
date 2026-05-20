"""Multi-suite / multi-model orchestration for ``audiobench run-matrix``.

Lets a single CLI invocation run several (suite, model) cells, write each
artifact to disk, and emit an aggregated summary that downstream tooling
(CI dashboards, leaderboards, JUnit converters) can consume.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


@dataclass
class MatrixCell:
    """One row of the matrix: which suite to run, against which model."""

    suite: str
    model: str
    seed: int | None = None
    limit: int | None = None
    conditions: list[str] | None = None
    pack: list[str] | None = None
    profile: str | None = None
    name: str | None = None

    def display_name(self) -> str:
        return self.name or f"{self.suite}::{self.model}"


@dataclass
class CellResult:
    cell: MatrixCell
    status: str  # "ok" | "error"
    duration_s: float
    output_path: str | None = None
    run_hash: str | None = None
    headline: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": asdict(self.cell),
            "status": self.status,
            "duration_s": round(self.duration_s, 4),
            "output_path": self.output_path,
            "run_hash": self.run_hash,
            "headline": self.headline,
            "error": self.error,
        }


@dataclass
class MatrixPlan:
    cells: list[MatrixCell]
    output_dir: Path
    seed: int = 1337
    gate_spec: dict[str, Any] | None = None


class MatrixConfigError(ValueError):
    """Raised when a matrix spec cannot be parsed."""


SUITE_GATE_KEYS: dict[str, str] = {
    "ab/asr-robust": "asr_robust",
    "ab/asr-hallucination": "asr_hallucination",
    "ab/sound-id": "sound_id",
}


def load_matrix_file(path: Path) -> MatrixPlan:
    """Parse a YAML/JSON matrix file into a :class:`MatrixPlan`."""
    if not path.exists():
        raise MatrixConfigError(f"matrix file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise MatrixConfigError(f"failed to parse matrix file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MatrixConfigError(f"matrix file must be a mapping, got {type(data).__name__}")

    raw_cells = data.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise MatrixConfigError("matrix file must include a non-empty 'cells' list")

    cells: list[MatrixCell] = []
    for index, entry in enumerate(raw_cells):
        if not isinstance(entry, dict):
            raise MatrixConfigError(f"cell #{index} must be a mapping")
        try:
            cells.append(_cell_from_dict(entry))
        except MatrixConfigError as exc:
            raise MatrixConfigError(f"cell #{index}: {exc}") from exc

    output_dir = Path(str(data.get("output_dir", "results/matrix")))
    seed = int(data.get("seed", 1337))
    gate_spec = data.get("gate") if isinstance(data.get("gate"), dict) else None
    return MatrixPlan(cells=cells, output_dir=output_dir, seed=seed, gate_spec=gate_spec)


def _cell_from_dict(entry: dict[str, Any]) -> MatrixCell:
    suite = entry.get("suite")
    model = entry.get("model")
    if not suite or not model:
        raise MatrixConfigError("each cell needs both 'suite' and 'model'")
    conditions = entry.get("conditions")
    if conditions is not None and not isinstance(conditions, list):
        raise MatrixConfigError("'conditions' must be a list of strings")
    pack = entry.get("pack")
    if pack is not None:
        if isinstance(pack, str):
            pack = [pack]
        if not isinstance(pack, list):
            raise MatrixConfigError("'pack' must be a list of strings or a single string")
    return MatrixCell(
        suite=str(suite),
        model=str(model),
        seed=int(entry["seed"]) if "seed" in entry else None,
        limit=int(entry["limit"]) if "limit" in entry else None,
        conditions=[str(item) for item in conditions] if conditions else None,
        pack=[str(item) for item in pack] if pack else None,
        profile=str(entry["profile"]) if entry.get("profile") else None,
        name=str(entry["name"]) if entry.get("name") else None,
    )


def build_cells_from_cli(
    suites: Iterable[str],
    models: Iterable[str],
    *,
    limit: int | None = None,
    conditions: list[str] | None = None,
    pack: list[str] | None = None,
    profile: str | None = None,
) -> list[MatrixCell]:
    """Build a cartesian set of cells from repeated CLI flags."""
    suites = [s for s in suites if s]
    models = [m for m in models if m]
    if not suites:
        raise MatrixConfigError("at least one --suite is required when no --matrix file is given")
    if not models:
        raise MatrixConfigError("at least one --model is required when no --matrix file is given")
    cells: list[MatrixCell] = []
    for suite in suites:
        for model in models:
            cells.append(
                MatrixCell(
                    suite=suite,
                    model=model,
                    limit=limit,
                    conditions=conditions,
                    pack=pack,
                    profile=profile,
                )
            )
    return cells


def _headline_for(result: dict[str, Any]) -> dict[str, Any]:
    suite = result.get("suite")
    if suite == "ab/asr-robust":
        return {
            "weighted_mean_wer": result.get("weighted_mean_wer"),
            "per_condition_wer": result.get("per_condition_wer"),
        }
    if suite == "ab/asr-hallucination":
        return {
            "weighted_hallucination_rate": result.get("weighted_hallucination_rate"),
            "non_speech_hallucination_rate": (result.get("headline") or {}).get(
                "non_speech_hallucination_rate"
            ),
            "top_finding_status": (result.get("top_finding") or {}).get("status"),
        }
    if suite == "ab/sound-id":
        headline = result.get("headline") or {}
        return {
            "weighted_recall": headline.get("weighted_recall"),
            "weighted_fpr": headline.get("weighted_fpr"),
            "components_understood": headline.get("components_understood"),
            "components_present": headline.get("components_present"),
        }
    if suite == "ab/fidelity-roundtrip":
        headline = result.get("headline") or {}
        return {
            "weighted_si_sdr_db": headline.get("weighted_si_sdr_db"),
            "max_true_peak_dbtp": headline.get("max_true_peak_dbtp"),
            "mean_loudness_delta_lu": headline.get("mean_loudness_delta_lu"),
        }
    if suite == "ab/psychoacoustic-masking":
        headline = result.get("headline") or {}
        return {
            "masking_respect_score": headline.get("masking_respect_score"),
            "respected_count": headline.get("respected_count"),
            "stimulus_count": headline.get("stimulus_count"),
        }
    if suite == "ab/phase-coherence":
        headline = result.get("headline") or {}
        return {
            "phase_coherence_score": headline.get("phase_coherence_score"),
            "mean_polarity_score": headline.get("mean_polarity_score"),
            "passed_count": headline.get("passed_count"),
            "stimulus_count": headline.get("stimulus_count"),
        }
    if suite == "ab/sed-urban":
        headline = result.get("headline") or {}
        return {
            "event_f1_iou50": headline.get("event_f1_iou50"),
            "segment_f1_1s": headline.get("segment_f1_1s"),
            "event_recall_iou50": headline.get("event_recall_iou50"),
            "clip_count": headline.get("clip_count"),
        }
    if suite == "ab/diarization-cw":
        headline = result.get("headline") or {}
        return {
            "der": headline.get("der"),
            "miss_rate": headline.get("miss_rate"),
            "false_alarm_rate": headline.get("false_alarm_rate"),
            "mean_speaker_count_error": headline.get("mean_speaker_count_error"),
        }
    return {}


def run_matrix(
    plan: MatrixPlan,
    *,
    runner: Callable[[MatrixCell, int], dict[str, Any]],
    output_dir: Path | None = None,
) -> list[CellResult]:
    """Execute every cell, write per-cell JSON, return :class:`CellResult` list."""
    target_dir = Path(output_dir) if output_dir is not None else plan.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    results: list[CellResult] = []
    for index, cell in enumerate(plan.cells):
        cell_seed = cell.seed if cell.seed is not None else plan.seed
        started = time.perf_counter()
        try:
            payload = runner(cell, cell_seed)
        except Exception as exc:  # noqa: BLE001 — we want to capture any error per cell
            duration = time.perf_counter() - started
            results.append(
                CellResult(
                    cell=cell,
                    status="error",
                    duration_s=duration,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        duration = time.perf_counter() - started

        run_hash = str(payload.get("run_hash", ""))
        slug = cell.display_name().replace("/", "__").replace("::", "__")
        file_name = f"{index:02d}-{slug}-{run_hash[:8] or 'run'}.json"
        output_path = target_dir / file_name
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        results.append(
            CellResult(
                cell=cell,
                status="ok",
                duration_s=duration,
                output_path=str(output_path),
                run_hash=run_hash or None,
                headline=_headline_for(payload),
            )
        )
    return results


def build_summary(
    plan: MatrixPlan,
    results: list[CellResult],
    *,
    gate_status: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-cell results into a single summary payload."""
    cells_out: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        entry = result.to_dict()
        entry["index"] = index
        if gate_status and result.cell.display_name() in gate_status:
            entry["gate"] = gate_status[result.cell.display_name()]
        cells_out.append(entry)

    ok_count = sum(1 for r in results if r.status == "ok")
    gate_failed = 0
    if gate_status:
        gate_failed = sum(1 for v in gate_status.values() if v.get("passed") is False)
    return {
        "schema": "audiobench.matrix.v1",
        "output_dir": str(plan.output_dir),
        "seed": plan.seed,
        "cell_count": len(results),
        "ok_count": ok_count,
        "error_count": len(results) - ok_count,
        "gate_failed_count": gate_failed,
        "cells": cells_out,
    }
