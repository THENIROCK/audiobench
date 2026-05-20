"""Threshold gating for run artifacts.

Used by ``audiobench gate`` to turn a run JSON into a pass/fail decision
that CI pipelines can act on through process exit codes.

Thresholds can be passed inline (CLI flags) or via a YAML/JSON file with a
suite-aware schema:

    asr_robust:
      max_weighted_mean_wer: 30.0
      max_wer:
        clean: 10.0
        noise-cafe-10db: 40.0
    asr_hallucination:
      max_weighted_hallucination_rate: 0.10
      max_non_speech_hallucination_rate: 0.15
    sound_id:
      min_weighted_recall: 0.60
      max_weighted_fpr: 0.10
      min_components_understood: 20
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


SUITE_KEY_ALIASES: dict[str, str] = {
    "ab/asr-robust": "asr_robust",
    "ab/asr-hallucination": "asr_hallucination",
    "ab/sound-id": "sound_id",
    "ab/fidelity-roundtrip": "fidelity_roundtrip",
    "ab/psychoacoustic-masking": "psychoacoustic_masking",
    "ab/phase-coherence": "phase_coherence",
    "ab/sed-urban": "sed_urban",
    "ab/diarization-cw": "diarization_cw",
}


@dataclass(frozen=True)
class GateCheck:
    """One pass/fail line item against a single threshold."""

    name: str
    actual: float | int | None
    threshold: float | int
    comparator: str  # "<=" (lower is better) or ">=" (higher is better)
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actual": self.actual,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class GateReport:
    suite: str
    model: str | None
    run_hash: str | None
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "model": self.model,
            "run_hash": self.run_hash,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "failure_count": len(self.failures),
        }


class GateConfigError(ValueError):
    """Raised when a gate spec is malformed or has no actionable thresholds."""


def load_thresholds(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON thresholds file."""
    if not path.exists():
        raise GateConfigError(f"thresholds file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise GateConfigError(f"failed to parse thresholds file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise GateConfigError(f"thresholds file must be a mapping, got {type(data).__name__}")
    return data


def merge_inline_thresholds(
    file_spec: dict[str, Any] | None,
    *,
    suite: str,
    inline: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI-provided thresholds onto a file spec for one suite.

    Inline values win when both are provided. Inline is a flat dict with
    the same leaf keys used in the file schema, scoped to ``suite``.
    """
    base = dict(file_spec or {})
    suite_key = SUITE_KEY_ALIASES.get(suite)
    section: dict[str, Any] = dict(base.get(suite_key, {})) if suite_key else {}
    for key, value in inline.items():
        if value is None:
            continue
        if key == "max_wer" and isinstance(value, dict):
            existing = dict(section.get("max_wer", {}))
            existing.update(value)
            section["max_wer"] = existing
        else:
            section[key] = value
    if suite_key is not None:
        base[suite_key] = section
    return base


def _check_lower(name: str, actual: Any, threshold: Any, *, detail: str = "") -> GateCheck:
    actual_val = None if actual is None else float(actual)
    threshold_val = float(threshold)
    passed = actual_val is not None and actual_val <= threshold_val
    return GateCheck(
        name=name,
        actual=actual_val,
        threshold=threshold_val,
        comparator="<=",
        passed=passed,
        detail=detail,
    )


def _check_higher(name: str, actual: Any, threshold: Any, *, detail: str = "") -> GateCheck:
    actual_val = None if actual is None else float(actual)
    threshold_val = float(threshold)
    passed = actual_val is not None and actual_val >= threshold_val
    return GateCheck(
        name=name,
        actual=actual_val,
        threshold=threshold_val,
        comparator=">=",
        passed=passed,
        detail=detail,
    )


def _section_for(suite: str, spec: dict[str, Any]) -> dict[str, Any]:
    key = SUITE_KEY_ALIASES.get(suite)
    if key is None:
        return {}
    section = spec.get(key) or {}
    if not isinstance(section, dict):
        raise GateConfigError(f"thresholds for {key!r} must be a mapping")
    return section


def evaluate_gate(run: dict[str, Any], spec: dict[str, Any]) -> GateReport:
    """Build a :class:`GateReport` from a run payload and threshold spec."""
    suite = str(run.get("suite", ""))
    if not suite:
        raise GateConfigError("run payload is missing 'suite'")
    section = _section_for(suite, spec)

    expected_suite = spec.get("suite")
    if expected_suite is not None and expected_suite != suite:
        raise GateConfigError(
            f"thresholds file targets suite={expected_suite!r} but run is suite={suite!r}"
        )

    report = GateReport(
        suite=suite,
        model=run.get("model"),
        run_hash=run.get("run_hash"),
    )

    if suite == "ab/asr-robust":
        _evaluate_asr_robust(run, section, report)
    elif suite == "ab/asr-hallucination":
        _evaluate_asr_hallucination(run, section, report)
    elif suite == "ab/sound-id":
        _evaluate_sound_id(run, section, report)
    elif suite == "ab/fidelity-roundtrip":
        _evaluate_fidelity(run, section, report)
    elif suite == "ab/psychoacoustic-masking":
        _evaluate_psychoacoustic(run, section, report)
    elif suite == "ab/phase-coherence":
        _evaluate_phase(run, section, report)
    elif suite == "ab/sed-urban":
        _evaluate_sed(run, section, report)
    elif suite == "ab/diarization-cw":
        _evaluate_diarization(run, section, report)
    else:
        raise GateConfigError(f"gate does not yet support suite {suite!r}")

    if not report.checks:
        raise GateConfigError(
            f"no thresholds applied to suite {suite!r}; add at least one to the spec"
        )
    return report


def _evaluate_asr_robust(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    weighted_max = section.get("max_weighted_mean_wer")
    if weighted_max is not None:
        report.checks.append(
            _check_lower(
                "weighted_mean_wer",
                run.get("weighted_mean_wer"),
                weighted_max,
            )
        )

    per_condition_caps = section.get("max_wer") or {}
    if not isinstance(per_condition_caps, dict):
        raise GateConfigError("asr_robust.max_wer must be a mapping of condition -> max")
    per_condition_wer = run.get("per_condition_wer", {}) or {}
    for condition, cap in per_condition_caps.items():
        actual = per_condition_wer.get(condition)
        report.checks.append(
            _check_lower(
                f"wer[{condition}]",
                actual,
                cap,
                detail="" if actual is not None else f"condition not in run (have: {sorted(per_condition_wer)})",
            )
        )


def _evaluate_asr_hallucination(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    weighted_max = section.get("max_weighted_hallucination_rate")
    if weighted_max is not None:
        report.checks.append(
            _check_lower(
                "weighted_hallucination_rate",
                run.get("weighted_hallucination_rate"),
                weighted_max,
            )
        )
    headline = run.get("headline") or {}
    overall_max = section.get("max_non_speech_hallucination_rate")
    if overall_max is not None:
        report.checks.append(
            _check_lower(
                "non_speech_hallucination_rate",
                headline.get("non_speech_hallucination_rate"),
                overall_max,
            )
        )
    per_domain = section.get("max_hallucination_rate") or {}
    if not isinstance(per_domain, dict):
        raise GateConfigError(
            "asr_hallucination.max_hallucination_rate must be a mapping of domain -> max"
        )
    per_condition_metrics = run.get("per_condition_metrics", {}) or {}
    for domain, cap in per_domain.items():
        metrics = per_condition_metrics.get(domain) or {}
        actual = metrics.get("non_speech_hallucination_rate")
        report.checks.append(
            _check_lower(
                f"hallucination_rate[{domain}]",
                actual,
                cap,
                detail="" if actual is not None else f"domain not in run (have: {sorted(per_condition_metrics)})",
            )
        )


def _evaluate_fidelity(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    min_sdr = section.get("min_weighted_si_sdr_db")
    if min_sdr is not None:
        report.checks.append(
            _check_higher(
                "weighted_si_sdr_db",
                headline.get("weighted_si_sdr_db"),
                min_sdr,
            )
        )
    max_tp = section.get("max_true_peak_dbtp")
    if max_tp is not None:
        report.checks.append(
            _check_lower(
                "max_true_peak_dbtp",
                headline.get("max_true_peak_dbtp"),
                max_tp,
            )
        )
    max_loudness = section.get("max_mean_loudness_delta_lu")
    if max_loudness is not None:
        actual = headline.get("mean_loudness_delta_lu")
        actual_abs = None if actual is None else abs(float(actual))
        report.checks.append(
            _check_lower(
                "abs_mean_loudness_delta_lu",
                actual_abs,
                max_loudness,
            )
        )


def _evaluate_psychoacoustic(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    min_score = section.get("min_masking_respect_score")
    if min_score is not None:
        report.checks.append(
            _check_higher(
                "masking_respect_score",
                headline.get("masking_respect_score"),
                min_score,
            )
        )
    max_inaudible = section.get("max_inaudible_energy_delta_db")
    if max_inaudible is not None:
        actual = headline.get("mean_inaudible_energy_delta_db")
        actual_abs = None if actual is None else abs(float(actual))
        report.checks.append(
            _check_lower(
                "abs_mean_inaudible_energy_delta_db",
                actual_abs,
                max_inaudible,
            )
        )


def _evaluate_phase(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    min_score = section.get("min_phase_coherence_score")
    if min_score is not None:
        report.checks.append(
            _check_higher(
                "phase_coherence_score",
                headline.get("phase_coherence_score"),
                min_score,
            )
        )
    min_polarity = section.get("min_mean_polarity_score")
    if min_polarity is not None:
        report.checks.append(
            _check_higher(
                "mean_polarity_score",
                headline.get("mean_polarity_score"),
                min_polarity,
            )
        )


def _evaluate_sed(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    min_event = section.get("min_event_f1")
    if min_event is not None:
        report.checks.append(
            _check_higher(
                "event_f1_iou50",
                headline.get("event_f1_iou50"),
                min_event,
            )
        )
    min_segment = section.get("min_segment_f1")
    if min_segment is not None:
        report.checks.append(
            _check_higher(
                "segment_f1_1s",
                headline.get("segment_f1_1s"),
                min_segment,
            )
        )
    min_recall = section.get("min_event_recall")
    if min_recall is not None:
        report.checks.append(
            _check_higher(
                "event_recall_iou50",
                headline.get("event_recall_iou50"),
                min_recall,
            )
        )


def _evaluate_diarization(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    max_der = section.get("max_der")
    if max_der is not None:
        report.checks.append(
            _check_lower(
                "der",
                headline.get("der"),
                max_der,
            )
        )
    max_speaker_err = section.get("max_speaker_count_error")
    if max_speaker_err is not None:
        report.checks.append(
            _check_lower(
                "mean_speaker_count_error",
                headline.get("mean_speaker_count_error"),
                max_speaker_err,
            )
        )
    max_miss = section.get("max_miss_rate")
    if max_miss is not None:
        report.checks.append(
            _check_lower(
                "miss_rate",
                headline.get("miss_rate"),
                max_miss,
            )
        )
    max_fa = section.get("max_false_alarm_rate")
    if max_fa is not None:
        report.checks.append(
            _check_lower(
                "false_alarm_rate",
                headline.get("false_alarm_rate"),
                max_fa,
            )
        )


def _evaluate_sound_id(
    run: dict[str, Any],
    section: dict[str, Any],
    report: GateReport,
) -> None:
    headline = run.get("headline") or {}
    min_recall = section.get("min_weighted_recall")
    if min_recall is not None:
        report.checks.append(
            _check_higher(
                "weighted_recall",
                headline.get("weighted_recall"),
                min_recall,
            )
        )
    max_fpr = section.get("max_weighted_fpr")
    if max_fpr is not None:
        report.checks.append(
            _check_lower(
                "weighted_fpr",
                headline.get("weighted_fpr"),
                max_fpr,
            )
        )
    min_components = section.get("min_components_understood")
    if min_components is not None:
        report.checks.append(
            _check_higher(
                "components_understood",
                headline.get("components_understood"),
                min_components,
            )
        )


def _escape_markup(text: str) -> str:
    """Escape ``[`` so Rich does not try to parse it as markup."""
    return text.replace("[", r"\[")


def format_console(report: GateReport) -> Iterable[str]:
    status = "PASS" if report.passed else "FAIL"
    head = (
        f"[bold]gate[/bold] · suite={report.suite}"
        f"{' · model=' + report.model if report.model else ''}"
        f"{' · run=' + report.run_hash[:8] if report.run_hash else ''}"
    )
    yield head
    for check in report.checks:
        marker = "[green]ok  [/green]" if check.passed else "[red]fail[/red]"
        actual = "—" if check.actual is None else f"{check.actual:g}"
        name = _escape_markup(check.name)
        detail = _escape_markup(check.detail) if check.detail else ""
        yield (
            f"  {marker} {name:32}  actual={actual:>8}  "
            f"{check.comparator} {check.threshold:g}"
            + (f"  ({detail})" if detail else "")
        )
    color = "green" if report.passed else "red"
    yield f"[{color}]{status}[/{color}]  {len(report.failures)} failing check(s)"
