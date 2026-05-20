"""Gradio Blocks app for ``audiobench --gui``.

The app reuses the same Python APIs the CLI calls, so behaviour stays in
lock-step with the terminal: same matrix runner, same gate evaluator,
same Rich renderers.
"""

from __future__ import annotations

import io
import json
import shlex
import time
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import yaml
from rich.console import Console

from audiobench import compare as compare_module
from audiobench import report as report_module
from audiobench.gating import (
    GateConfigError,
    evaluate_gate,
    format_console as format_gate_console,
    load_thresholds,
)
from audiobench.matrix import (
    MatrixCell,
    MatrixConfigError,
    MatrixPlan,
    build_summary,
    load_matrix_file,
    run_matrix as run_matrix_cells,
)
from audiobench.models.asr_registry import list_models as list_asr_models
from audiobench.models.registry import list_models as list_sound_id_models
from audiobench.models.signal_registry import list_models as list_signal_models
from audiobench.suites import (
    asr_suite_ids,
    list_suite_specs,
    signal_suite_ids,
    sound_id as sound_id_suite,
)


# ---------------------------------------------------------------------------
# Constants reused across tabs
# ---------------------------------------------------------------------------


RUNNABLE_SUITES: list[str] = sorted(
    spec.suite_id for spec in list_suite_specs() if spec.runnable
)

CELL_COLUMNS: list[str] = [
    "name",
    "suite",
    "model",
    "seed",
    "limit",
    "conditions",
    "pack",
    "profile",
]

# Per-suite gate threshold keys we surface in the GUI editor. Mirrors the
# fields handled by ``audiobench.gating``.
GATE_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    "asr_robust": [
        ("max_weighted_mean_wer", "max weighted mean WER", "lower"),
    ],
    "asr_hallucination": [
        ("max_weighted_hallucination_rate", "max weighted hallucination rate", "lower"),
        ("max_non_speech_hallucination_rate", "max non-speech hallucination rate", "lower"),
    ],
    "sound_id": [
        ("min_weighted_recall", "min weighted recall", "higher"),
        ("max_weighted_fpr", "max weighted FPR", "lower"),
        ("min_components_understood", "min components understood", "higher"),
    ],
    "fidelity_roundtrip": [
        ("min_weighted_si_sdr_db", "min weighted SI-SDR (dB)", "higher"),
        ("max_true_peak_dbtp", "max true peak (dBTP)", "lower"),
        ("max_mean_loudness_delta_lu", "max |loudness delta| (LU)", "lower"),
    ],
    "psychoacoustic_masking": [
        ("min_masking_respect_score", "min masking-respect score", "higher"),
        ("max_inaudible_energy_delta_db", "max |inaudible energy delta| (dB)", "lower"),
    ],
    "phase_coherence": [
        ("min_phase_coherence_score", "min phase coherence score", "higher"),
        ("min_mean_polarity_score", "min mean polarity score", "higher"),
    ],
}

GATE_KEY_TO_SUITE: dict[str, str] = {
    "asr_robust": "ab/asr-robust",
    "asr_hallucination": "ab/asr-hallucination",
    "sound_id": "ab/sound-id",
    "fidelity_roundtrip": "ab/fidelity-roundtrip",
    "psychoacoustic_masking": "ab/psychoacoustic-masking",
    "phase_coherence": "ab/phase-coherence",
}


# ---------------------------------------------------------------------------
# Rich -> HTML helper
# ---------------------------------------------------------------------------


def _rich_html(render: Callable[[Console], None]) -> str:
    """Run ``render(console)`` against a recording console and return HTML."""
    buf = io.StringIO()
    console = Console(record=True, file=buf, width=120, force_terminal=False)
    try:
        render(console)
    except Exception as exc:  # noqa: BLE001 — surface render errors in the GUI
        return f"<pre style='color:#c33'>render error: {type(exc).__name__}: {exc}</pre>"
    html = console.export_html(inline_styles=True, code_format="<pre style='font-family:JetBrains Mono,monospace;'>{code}</pre>")
    return html


def _rich_lines_html(lines: list[str]) -> str:
    def render(console: Console) -> None:
        for line in lines:
            console.print(line)

    return _rich_html(render)


# ---------------------------------------------------------------------------
# Model dropdown helpers
# ---------------------------------------------------------------------------


# `list_asr_models()` returns ``"whisper-*"`` as a checkpoint pattern. Expand
# it into concrete picks so the GUI dropdown is usable; the dropdown also has
# ``allow_custom_value=True`` so users can type any other whisper checkpoint
# (or a plugin-registered adapter id).
ASR_MODEL_SUGGESTIONS: list[str] = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v2",
    "whisper-large-v3",
]


def models_for_suite(suite_id: str | None) -> list[str]:
    if not suite_id:
        return []
    if suite_id in asr_suite_ids():
        concrete = [m for m in list_asr_models() if m != "whisper-*"]
        return sorted({*concrete, *ASR_MODEL_SUGGESTIONS})
    if suite_id == sound_id_suite.SUITE_ID:
        return sorted(list_sound_id_models())
    if suite_id in signal_suite_ids():
        return sorted(list_signal_models())
    return []


def _dataframe_to_rows(value) -> list[list]:
    """Normalize a ``gr.Dataframe`` value into a plain list of rows.

    Gradio may hand us a ``pandas.DataFrame``, a list of lists, or ``None``
    depending on the component state. We always work with plain lists.
    """
    if value is None:
        return []
    # pandas.DataFrame
    if hasattr(value, "values") and hasattr(value, "columns"):
        return value.values.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []


# ---------------------------------------------------------------------------
# Cell <-> table conversions
# ---------------------------------------------------------------------------


def _csv(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _csv_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cells_to_rows(cells: list[MatrixCell]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for cell in cells:
        rows.append(
            [
                cell.name or "",
                cell.suite,
                cell.model,
                cell.seed if cell.seed is not None else "",
                cell.limit if cell.limit is not None else "",
                _csv(cell.conditions),
                _csv(cell.pack),
                cell.profile or "",
            ]
        )
    return rows


def rows_to_cells(rows: Any) -> list[MatrixCell]:
    rows = _dataframe_to_rows(rows)
    if not rows:
        return []
    cells: list[MatrixCell] = []
    for row in rows:
        if not row or not any(str(x).strip() for x in row if x is not None):
            continue
        # Pad short rows defensively.
        padded = list(row) + [""] * (len(CELL_COLUMNS) - len(row))
        name, suite, model, seed, limit, conditions, pack, profile = padded[:8]
        suite_str = str(suite).strip()
        model_str = str(model).strip()
        if not suite_str or not model_str:
            continue
        cells.append(
            MatrixCell(
                suite=suite_str,
                model=model_str,
                seed=_int_or_none(seed),
                limit=_int_or_none(limit),
                conditions=_csv_list(conditions),
                pack=_csv_list(pack),
                profile=str(profile).strip() or None,
                name=str(name).strip() or None,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# Matrix plan <-> YAML
# ---------------------------------------------------------------------------


def _cell_to_yaml_dict(cell: MatrixCell) -> dict[str, Any]:
    entry: dict[str, Any] = {"suite": cell.suite, "model": cell.model}
    if cell.name:
        entry["name"] = cell.name
    if cell.seed is not None:
        entry["seed"] = cell.seed
    if cell.limit is not None:
        entry["limit"] = cell.limit
    if cell.conditions:
        entry["conditions"] = list(cell.conditions)
    if cell.pack:
        entry["pack"] = list(cell.pack)
    if cell.profile:
        entry["profile"] = cell.profile
    return entry


def build_matrix_yaml(
    *,
    output_dir: str,
    seed: int,
    cells: list[MatrixCell],
    gate_spec: dict[str, Any] | None,
) -> str:
    plan: dict[str, Any] = {
        "output_dir": output_dir or "results/matrix",
        "seed": int(seed) if seed is not None else 1337,
        "cells": [_cell_to_yaml_dict(c) for c in cells],
    }
    if gate_spec:
        # Drop empty sections to keep the YAML tidy.
        clean: dict[str, Any] = {}
        for key, value in gate_spec.items():
            if isinstance(value, dict) and value:
                clean[key] = value
        if clean:
            plan["gate"] = clean
    return yaml.safe_dump(plan, sort_keys=False, default_flow_style=False)


def gate_state_to_spec(gate_state: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    """Drop ``None`` values; return only suites with at least one threshold."""
    if not gate_state:
        return {}
    spec: dict[str, Any] = {}
    for suite_key, fields in gate_state.items():
        cleaned: dict[str, Any] = {}
        for key, value in (fields or {}).items():
            if value is None or value == "":
                continue
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
        if cleaned:
            spec[suite_key] = cleaned
    return spec


def spec_to_gate_state(spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {key: {} for key in GATE_FIELDS}
    for suite_key, fields in (spec or {}).items():
        if suite_key not in GATE_FIELDS or not isinstance(fields, dict):
            continue
        for field_key, _, _ in GATE_FIELDS[suite_key]:
            value = fields.get(field_key)
            if isinstance(value, (int, float)):
                state[suite_key][field_key] = float(value)
    return state


# ---------------------------------------------------------------------------
# Session / result discovery
# ---------------------------------------------------------------------------


def discover_sessions(results_root: str) -> list[list[Any]]:
    """Return a table of (name, cells, ok, error, gate, modified) for sessions."""
    root = Path(results_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    rows: list[list[Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            [
                child.name,
                summary.get("cell_count", 0),
                summary.get("ok_count", 0),
                summary.get("error_count", 0),
                summary.get("gate_failed_count", 0),
                time.strftime("%Y-%m-%d %H:%M", time.localtime(summary_path.stat().st_mtime)),
                str(summary_path),
            ]
        )
    return rows


def discover_loose_runs(results_root: str) -> list[list[Any]]:
    root = Path(results_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    rows: list[list[Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        suite = data.get("suite", "")
        model = data.get("model", "")
        run_hash = (data.get("run_hash") or "")[:8]
        rows.append(
            [
                path.name,
                suite,
                model,
                run_hash,
                time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
                str(path),
            ]
        )
    return rows


def render_summary_html(summary_path: str) -> str:
    if not summary_path:
        return ""
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"<pre style='color:#c33'>failed to read summary: {exc}</pre>"

    return _rich_html(lambda console: _render_matrix_console_with(summary, console))


def _render_matrix_console_with(summary: dict[str, Any], console: Console) -> None:
    """Wrap cli._render_matrix_console to accept an explicit console."""
    from audiobench import cli as cli_module

    original = cli_module.console
    cli_module.console = console
    try:
        cli_module._render_matrix_console(summary)
    finally:
        cli_module.console = original


def list_cells_for_session(summary_path: str) -> list[list[Any]]:
    if not summary_path:
        return []
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[list[Any]] = []
    for entry in summary.get("cells", []):
        cell = entry.get("cell") or {}
        name = cell.get("name") or f"{cell.get('suite')}::{cell.get('model')}"
        rows.append(
            [
                entry.get("index", 0),
                name,
                entry.get("status", ""),
                entry.get("run_hash", "") or "",
                entry.get("output_path", "") or "",
            ]
        )
    return rows


def render_run_summary_html(run_path: str) -> str:
    if not run_path:
        return ""
    p = Path(run_path)
    if not p.exists():
        return f"<pre style='color:#c33'>file not found: {run_path}</pre>"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"<pre style='color:#c33'>invalid JSON: {exc}</pre>"
    return _rich_html(lambda c: report_module.render_run_summary(data, console=c))


# ---------------------------------------------------------------------------
# Builder runtime — execute a matrix using the same code path as the CLI
# ---------------------------------------------------------------------------


def run_builder_matrix(
    *,
    output_dir: str,
    seed: int,
    cells: list[MatrixCell],
    gate_spec: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    """Execute the matrix and return (html_log, summary)."""
    if not cells:
        return ("<pre style='color:#c33'>no cells to run; add at least one row.</pre>", None)

    from audiobench.cli import _gate_cell, _run_cell

    plan = MatrixPlan(
        cells=cells,
        output_dir=Path(output_dir or "results/matrix"),
        seed=int(seed) if seed is not None else 1337,
        gate_spec=gate_spec or None,
    )

    try:
        results = run_matrix_cells(plan, runner=_run_cell)
    except Exception as exc:  # noqa: BLE001 — surface to GUI
        return (f"<pre style='color:#c33'>run failed: {type(exc).__name__}: {exc}</pre>", None)

    gate_status: dict[str, dict[str, Any]] = {}
    if gate_spec:
        for result in results:
            report = _gate_cell(result, gate_spec)
            if report is not None:
                gate_status[result.cell.display_name()] = report

    summary = build_summary(plan, results, gate_status=gate_status)
    summary_path = plan.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    html = _rich_html(lambda c: _render_matrix_console_with(summary, c))
    html += (
        f"<p style='font-family:JetBrains Mono,monospace;'>wrote: "
        f"<code>{summary_path}</code></p>"
    )
    return (html, summary)


# ---------------------------------------------------------------------------
# Secondary tabs — run / compare / inspect / gate / push
# ---------------------------------------------------------------------------


def run_single_suite(
    suite_id: str,
    model: str,
    output: str,
    seed: int,
    limit: str,
    conditions: str,
    pack: str,
    profile: str,
) -> tuple[str, str]:
    """Mirror ``audiobench run`` for the GUI Run tab."""
    from audiobench import cli as cli_module
    from audiobench.cli import (
        ASR_HALLUCINATION_SUITE_ID,
        ASR_SUITE_ID,
        FIDELITY_SUITE_ID,
        PHASE_SUITE_ID,
        PSYCHO_SUITE_ID,
        run_asr_hallucination_suite,
        run_asr_suite,
        run_fidelity_suite,
        run_phase_suite,
        run_psycho_suite,
    )
    from audiobench.prompts import load_prompts

    limit_v = _int_or_none(limit)
    conditions_v = _csv_list(conditions)
    pack_v = _csv_list(pack)
    profile_v = (profile or "").strip() or None

    try:
        if suite_id in asr_suite_ids():
            if suite_id == ASR_SUITE_ID:
                result = run_asr_suite(
                    model_name=model,
                    seed=int(seed),
                    limit=limit_v,
                    condition_names=conditions_v,
                )
            else:
                result = run_asr_hallucination_suite(
                    model_name=model,
                    seed=int(seed),
                    limit=limit_v,
                    condition_names=conditions_v,
                )
        elif suite_id == sound_id_suite.SUITE_ID:
            prompt_spec = load_prompts(None)
            result = sound_id_suite.run_suite(
                model_name=model,
                seed=int(seed),
                pack_ids=pack_v,
                selected_conditions=conditions_v,
                profile_name=profile_v,
                custom_mixtures=None,
                limit=limit_v,
                prompt_spec=prompt_spec,
                prompt_ensemble=None,
            )
        elif suite_id in signal_suite_ids():
            if suite_id == FIDELITY_SUITE_ID:
                result = run_fidelity_suite(
                    model_name=model,
                    seed=int(seed),
                    limit=limit_v,
                    condition_names=conditions_v,
                )
            elif suite_id == PSYCHO_SUITE_ID:
                result = run_psycho_suite(model_name=model, seed=int(seed), limit=limit_v)
            else:
                result = run_phase_suite(model_name=model, seed=int(seed), limit=limit_v)
        else:
            return (f"<pre style='color:#c33'>unknown suite: {suite_id}</pre>", "")
    except Exception as exc:  # noqa: BLE001
        return (f"<pre style='color:#c33'>{type(exc).__name__}: {exc}</pre>", "")

    out_path = Path(output) if output else Path("results") / f"run-{result['run_hash'][:8]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    html = _rich_html(lambda c: report_module.render_run_summary(result, console=c))
    html += f"<p style='font-family:JetBrains Mono,monospace;'>wrote: <code>{out_path}</code></p>"
    return (html, str(out_path))


def compare_runs(run_a: str, run_b: str, allow_mismatched_prompt: bool) -> str:
    if not run_a or not run_b:
        return "<pre>provide both run files.</pre>"
    if not Path(run_a).exists() or not Path(run_b).exists():
        return "<pre style='color:#c33'>one or both files do not exist.</pre>"
    try:
        left = json.loads(Path(run_a).read_text(encoding="utf-8"))
        right = json.loads(Path(run_b).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"<pre style='color:#c33'>invalid JSON: {exc}</pre>"
    try:
        return _rich_html(
            lambda c: compare_module.render_run_pair(
                left, right, console=c, allow_mismatched_prompt=allow_mismatched_prompt
            )
        )
    except compare_module.CompareMismatchError as exc:
        return f"<pre style='color:#c33'>{exc}</pre>"


def inspect_run(run_path: str, index_kind: str, index_1based: int) -> str:
    if not run_path:
        return ""
    p = Path(run_path)
    if not p.exists():
        return f"<pre style='color:#c33'>file not found: {run_path}</pre>"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"<pre style='color:#c33'>invalid JSON: {exc}</pre>"

    suite = data.get("suite")
    from audiobench import cli as cli_module

    def _render(console: Console) -> None:
        original = cli_module.console
        cli_module.console = console
        try:
            if index_kind == "clip":
                cli_module._inspect_asr(data, int(index_1based))
            else:
                # Reuse the sound-id branch of cli.inspect.
                from audiobench.suites import sound_id as sound_id_mod

                if suite != sound_id_mod.SUITE_ID:
                    console.print(f"[red]inspect (mixture) only applies to {sound_id_mod.SUITE_ID}[/red]")
                    return
                per_mixture = data.get("per_mixture", [])
                if not per_mixture:
                    console.print("[red]run JSON has no per_mixture records[/red]")
                    return
                mixture = int(index_1based)
                if mixture < 1 or mixture > len(per_mixture):
                    console.print(f"[red]mixture index out of range (1..{len(per_mixture)})[/red]")
                    return
                record = per_mixture[mixture - 1]
                console.print(
                    f"mixture {mixture} (pack={record['pack']}, condition={record['condition']}, "
                    f"name={record['mixture_name']})"
                )
                components = record.get("components_present", [])
                console.print(f"  ground truth: {', '.join(components)}")
                for probe in record.get("probes", []):
                    ans = "yes" if probe.get("answered_yes") else "no "
                    expected = probe.get("expected")
                    marker = "[green]✓[/green]" if (bool(probe.get("answered_yes")) == bool(expected)) else "[red]✗[/red]"
                    console.print(f"  {marker} {probe.get('label'):20} -> {ans} (expected={'yes' if expected else 'no'})")
        finally:
            cli_module.console = original

    try:
        return _rich_html(_render)
    except Exception as exc:  # noqa: BLE001
        return f"<pre style='color:#c33'>{type(exc).__name__}: {exc}</pre>"


def evaluate_gate_path(run_path: str, gate_yaml: str) -> tuple[str, str]:
    if not run_path:
        return ("", "")
    p = Path(run_path)
    if not p.exists():
        return (f"<pre style='color:#c33'>file not found: {run_path}</pre>", "")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (f"<pre style='color:#c33'>invalid JSON: {exc}</pre>", "")
    try:
        spec = yaml.safe_load(gate_yaml) if gate_yaml.strip() else {}
    except yaml.YAMLError as exc:
        return (f"<pre style='color:#c33'>invalid gate YAML: {exc}</pre>", "")
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        return ("<pre style='color:#c33'>gate spec must be a mapping</pre>", "")
    try:
        report = evaluate_gate(payload, spec)
    except GateConfigError as exc:
        return (f"<pre style='color:#c33'>{exc}</pre>", "")
    lines = list(format_gate_console(report))
    html = _rich_lines_html(lines)
    return (html, json.dumps(report.to_dict(), indent=2, sort_keys=True))


def push_run(
    run_path: str,
    repo: str,
    token: str,
    notes: str,
    tags: str,
    dry_run: bool,
    overwrite: bool,
    private: bool,
) -> str:
    if not run_path:
        return ""
    p = Path(run_path)
    if not p.exists():
        return f"<pre style='color:#c33'>file not found: {run_path}</pre>"
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"<pre style='color:#c33'>invalid JSON: {exc}</pre>"

    from audiobench.leaderboard import (
        build_submission_record,
        default_dataset_repo_id,
        push_submission,
    )

    submission_tags = _csv_list(tags)
    repo_id = repo.strip() or None
    if repo_id is None:
        try:
            repo_id = default_dataset_repo_id(token=token or None)
        except RuntimeError:
            repo_id = None

    if dry_run:
        try:
            record = build_submission_record(
                payload,
                notes=notes or None,
                tags=submission_tags,
                source_file=str(p),
            )
        except ValueError as exc:
            return f"<pre style='color:#c33'>{exc}</pre>"
        preview = {
            "suite": record["suite"],
            "model": record["model"],
            "run_hash": record["run_hash"],
            "mode": "dry-run",
            "repo": repo_id,
            "path_preview": f"submissions/{record['suite'].replace('/', '__')}/{record['run_hash']}.json",
        }
        return f"<pre>{json.dumps(preview, indent=2)}</pre>"

    if repo_id is None:
        return "<pre style='color:#c33'>no repo provided and no default could be resolved (hf auth login?)</pre>"
    try:
        record, result = push_submission(
            payload,
            repo_id=repo_id,
            token=token or None,
            private=private,
            allow_overwrite=overwrite,
            notes=notes or None,
            tags=submission_tags,
            source_file=str(p),
        )
    except Exception as exc:  # noqa: BLE001
        return f"<pre style='color:#c33'>push failed: {exc}</pre>"
    return (
        "<pre>"
        + json.dumps(
            {
                "repo": result.repo_id,
                "path_in_repo": result.path_in_repo,
                "uploaded": result.uploaded,
                "duplicate": result.duplicate,
                "dataset_url": result.dataset_url,
            },
            indent=2,
        )
        + "</pre>"
    )


# ---------------------------------------------------------------------------
# Equivalent CLI command renderers (so users can copy/paste)
# ---------------------------------------------------------------------------


def cli_for_run(suite_id, model, output, seed, limit, conditions, pack, profile) -> str:
    parts = ["audiobench", "run", suite_id or "<suite>", "--model", model or "<model>"]
    if output:
        parts += ["--output", output]
    if seed:
        parts += ["--seed", str(seed)]
    if limit:
        parts += ["--limit", str(limit)]
    if conditions:
        parts += ["--conditions", conditions]
    if pack:
        parts += ["--pack", pack]
    if profile:
        parts += ["--profile", profile]
    return shlex.join(str(x) for x in parts)


def cli_for_compare(a, b, allow) -> str:
    parts = ["audiobench", "compare", a or "<run-a.json>", b or "<run-b.json>"]
    if allow:
        parts.append("--allow-mismatched-prompt")
    return shlex.join(parts)


def cli_for_inspect(run_path, kind, index) -> str:
    parts = ["audiobench", "inspect", run_path or "<run.json>", f"--{kind}", str(index)]
    return shlex.join(parts)


def cli_for_gate(run_path) -> str:
    return shlex.join(["audiobench", "gate", run_path or "<run.json>", "--thresholds", "gate.yaml"])


def cli_for_push(run_path, repo, dry_run) -> str:
    parts = ["audiobench", "push", run_path or "<run.json>"]
    if repo:
        parts += ["--repo", repo]
    if dry_run:
        parts.append("--dry-run")
    return shlex.join(parts)


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


INTRO_MD = """
# audiobench

Local GUI for structuring and reviewing audio ML benchmarks. The two main tabs
are **Test Builder** (compose a `matrix.yaml`-style session) and **Results**
(browse sessions and drill into runs). Everything here can also be done from
the terminal with `audiobench <command>`.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="audiobench", theme=gr.themes.Soft()) as app:
        gr.Markdown(INTRO_MD)

        # State: matrix builder.
        cells_state = gr.State([])  # list[MatrixCell]
        gate_state = gr.State({key: {} for key in GATE_FIELDS})  # dict[str, dict[str, float]]

        with gr.Tabs():
            _builder_tab(cells_state, gate_state)
            _results_tab()
            _run_tab()
            _compare_tab()
            _inspect_tab()
            _gate_tab()
            _push_tab()

    return app


# --- Builder tab -----------------------------------------------------------


def _builder_tab(cells_state: gr.State, gate_state: gr.State) -> None:
    with gr.Tab("Test Builder"):
        gr.Markdown(
            "Compose a multi-cell test session. Each row is one `(suite, model)` "
            "cell — equivalent to `audiobench run-matrix --matrix matrix.yaml`."
        )

        with gr.Row():
            output_dir = gr.Textbox(value="results/matrix", label="output_dir")
            seed_box = gr.Number(value=1337, precision=0, label="seed")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Cells")
                cells_df = gr.Dataframe(
                    headers=CELL_COLUMNS,
                    datatype=["str", "str", "str", "number", "number", "str", "str", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(len(CELL_COLUMNS), "fixed"),
                    interactive=True,
                    wrap=True,
                )

                with gr.Row():
                    _default_suite = RUNNABLE_SUITES[0] if RUNNABLE_SUITES else None
                    _default_models = models_for_suite(_default_suite)
                    suite_pick = gr.Dropdown(
                        choices=RUNNABLE_SUITES,
                        label="suite",
                        value=_default_suite,
                    )
                    model_pick = gr.Dropdown(
                        choices=_default_models,
                        value=_default_models[0] if _default_models else None,
                        label="model",
                        allow_custom_value=True,
                    )
                    add_btn = gr.Button("Add cell", variant="primary")
                    clear_btn = gr.Button("Clear cells")

                def _refresh_models(suite_id: str | None):
                    options = models_for_suite(suite_id)
                    return gr.update(choices=options, value=options[0] if options else None)

                suite_pick.change(
                    fn=_refresh_models,
                    inputs=suite_pick,
                    outputs=model_pick,
                )

                gr.Markdown(
                    "### Gate thresholds (optional)\n"
                    "Pass/fail caps applied to each cell after it runs (same as "
                    "`audiobench gate`). Leave blank to skip gating; fill any value "
                    "and the YAML preview will gain a `gate:` section."
                )
                gate_inputs: dict[tuple[str, str], gr.Number] = {}
                for suite_key, fields in GATE_FIELDS.items():
                    with gr.Accordion(GATE_KEY_TO_SUITE[suite_key], open=False):
                        for field_key, label, direction in fields:
                            box = gr.Number(
                                label=f"{label}  ({direction} is better)",
                                value=None,
                            )
                            gate_inputs[(suite_key, field_key)] = box

            with gr.Column(scale=2):
                gr.Markdown("### matrix.yaml preview")
                yaml_preview = gr.Code(language="yaml", value="", lines=18, label="matrix.yaml")

                with gr.Row():
                    save_path = gr.Textbox(value="matrix.yaml", label="save path", scale=3)
                    save_btn = gr.Button("Save", scale=1)
                save_status = gr.Markdown("")

                upload = gr.File(label="Load matrix.yaml", file_types=[".yaml", ".yml", ".json"])

                gr.Markdown("### Run")
                run_btn = gr.Button("Run matrix", variant="primary")
                run_log = gr.HTML("")

        # --- Helpers wired to UI events ---

        def _sync_yaml(cells_rows, out_dir, seed, *gate_values) -> tuple[str, list[MatrixCell], dict[str, dict[str, float]]]:
            cells = rows_to_cells(cells_rows)
            new_gate: dict[str, dict[str, float]] = {key: {} for key in GATE_FIELDS}
            for (suite_key, field_key), value in zip(gate_inputs.keys(), gate_values):
                if value is None or value == "":
                    continue
                try:
                    new_gate[suite_key][field_key] = float(value)
                except (TypeError, ValueError):
                    continue
            cleaned = {k: v for k, v in new_gate.items() if v}
            yaml_text = build_matrix_yaml(
                output_dir=out_dir, seed=int(seed or 1337), cells=cells, gate_spec=cleaned
            )
            return yaml_text, cells, new_gate

        sync_inputs = [cells_df, output_dir, seed_box, *gate_inputs.values()]
        sync_outputs = [yaml_preview, cells_state, gate_state]

        for component in sync_inputs:
            event = getattr(component, "change", None)
            if event is None:
                continue
            event(fn=_sync_yaml, inputs=sync_inputs, outputs=sync_outputs)

        def _add_cell(rows, suite, model):
            if not suite or not model:
                return rows
            current = rows_to_cells(rows)
            current.append(MatrixCell(suite=suite, model=model))
            return cells_to_rows(current)

        add_btn.click(fn=_add_cell, inputs=[cells_df, suite_pick, model_pick], outputs=cells_df)
        clear_btn.click(fn=lambda: [], inputs=None, outputs=cells_df)

        def _save_yaml(path_str, yaml_text):
            if not path_str:
                return "no path given."
            path = Path(path_str).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml_text, encoding="utf-8")
            except OSError as exc:
                return f"failed: {exc}"
            return f"wrote `{path}`."

        save_btn.click(fn=_save_yaml, inputs=[save_path, yaml_preview], outputs=save_status)

        def _load_yaml(file) -> tuple[list[list[Any]], str, int, dict[str, dict[str, float]], list[gr.Number]]:
            if file is None:
                return ([], "results/matrix", 1337, {key: {} for key in GATE_FIELDS}, *[gr.update() for _ in gate_inputs])  # type: ignore[return-value]
            try:
                plan = load_matrix_file(Path(file.name if hasattr(file, "name") else file))
            except MatrixConfigError as exc:
                # Surface error via save_status; leave state alone.
                # Without a clean signal channel, just no-op.
                return ([], "results/matrix", 1337, {key: {} for key in GATE_FIELDS}, *[gr.update() for _ in gate_inputs])  # type: ignore[return-value]
            rows = cells_to_rows(plan.cells)
            gate = spec_to_gate_state(plan.gate_spec or {})
            updates = []
            for (suite_key, field_key) in gate_inputs.keys():
                updates.append(gr.update(value=gate.get(suite_key, {}).get(field_key)))
            return rows, str(plan.output_dir), plan.seed, gate, *updates  # type: ignore[return-value]

        upload.change(
            fn=_load_yaml,
            inputs=upload,
            outputs=[cells_df, output_dir, seed_box, gate_state, *gate_inputs.values()],
        )

        def _run(cells_rows, out_dir, seed, gate_values_state):
            cells = rows_to_cells(cells_rows)
            spec = gate_state_to_spec(gate_values_state)
            html, _summary = run_builder_matrix(
                output_dir=out_dir or "results/matrix",
                seed=int(seed or 1337),
                cells=cells,
                gate_spec=spec or None,
            )
            return html

        run_btn.click(fn=_run, inputs=[cells_df, output_dir, seed_box, gate_state], outputs=run_log)


# --- Results tab -----------------------------------------------------------


def _results_tab() -> None:
    with gr.Tab("Results"):
        gr.Markdown(
            "Browse testing sessions (any directory under `results/` containing "
            "`summary.json`) and ad-hoc run JSON files."
        )

        with gr.Row():
            root_box = gr.Textbox(value="results", label="results root")
            refresh_btn = gr.Button("Refresh")

        gr.Markdown("### Sessions")
        sessions_df = gr.Dataframe(
            headers=["session", "cells", "ok", "error", "gate_failed", "modified", "path"],
            datatype=["str", "number", "number", "number", "number", "str", "str"],
            interactive=False,
            wrap=True,
        )

        session_summary = gr.HTML("")

        gr.Markdown("### Cells in selected session")
        cells_df = gr.Dataframe(
            headers=["#", "name", "status", "run_hash", "run_path"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### Ad-hoc runs (top-level `*.json`)")
        loose_df = gr.Dataframe(
            headers=["file", "suite", "model", "run_hash", "modified", "path"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### Run detail")
        run_html = gr.HTML("")
        run_path_box = gr.Textbox(label="run JSON path", placeholder="paste a run path or click a row above")
        load_run_btn = gr.Button("Load run detail")

        def _refresh(root):
            return discover_sessions(root), discover_loose_runs(root)

        refresh_btn.click(fn=_refresh, inputs=root_box, outputs=[sessions_df, loose_df])

        def _row_path(rows, evt: gr.SelectData) -> str | None:
            data = _dataframe_to_rows(rows)
            try:
                idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                return data[idx][-1]
            except (IndexError, TypeError):
                return None

        def _select_session(rows, evt: gr.SelectData):
            path = _row_path(rows, evt)
            if not path:
                return "", []
            return render_summary_html(path), list_cells_for_session(path)

        sessions_df.select(fn=_select_session, inputs=sessions_df, outputs=[session_summary, cells_df])

        def _select_cell(rows, evt: gr.SelectData):
            path = _row_path(rows, evt)
            if not path:
                return "", ""
            return render_run_summary_html(path), path

        cells_df.select(fn=_select_cell, inputs=cells_df, outputs=[run_html, run_path_box])

        def _select_loose(rows, evt: gr.SelectData):
            path = _row_path(rows, evt)
            if not path:
                return "", ""
            return render_run_summary_html(path), path

        loose_df.select(fn=_select_loose, inputs=loose_df, outputs=[run_html, run_path_box])

        load_run_btn.click(fn=render_run_summary_html, inputs=run_path_box, outputs=run_html)


# --- Run tab ---------------------------------------------------------------


def _run_tab() -> None:
    with gr.Tab("Run"):
        gr.Markdown("Run a single `(suite, model)` cell — equivalent to `audiobench run`.")
        _default_suite = RUNNABLE_SUITES[0] if RUNNABLE_SUITES else None
        _default_models = models_for_suite(_default_suite)
        with gr.Row():
            suite = gr.Dropdown(choices=RUNNABLE_SUITES, value=_default_suite, label="suite")
            model = gr.Dropdown(
                choices=_default_models,
                value=_default_models[0] if _default_models else None,
                allow_custom_value=True,
                label="model",
            )

        def _refresh_models(suite_id: str | None):
            options = models_for_suite(suite_id)
            return gr.update(choices=options, value=options[0] if options else None)

        suite.change(fn=_refresh_models, inputs=suite, outputs=model)

        with gr.Row():
            output = gr.Textbox(label="output path (optional)")
            seed = gr.Number(value=1337, precision=0, label="seed")
            limit = gr.Textbox(label="limit", placeholder="(int or blank)")
        with gr.Row():
            conditions = gr.Textbox(label="conditions (comma)")
            pack = gr.Textbox(label="pack (sound-id only, comma)")
            profile = gr.Textbox(label="profile (sound-id only)")

        cli_box = gr.Code(language="shell", label="equivalent CLI", value="")
        for c in (suite, model, output, seed, limit, conditions, pack, profile):
            c.change(
                fn=cli_for_run,
                inputs=[suite, model, output, seed, limit, conditions, pack, profile],
                outputs=cli_box,
            )

        run_btn = gr.Button("Run", variant="primary")
        html_out = gr.HTML("")
        path_out = gr.Textbox(label="wrote", interactive=False)
        run_btn.click(
            fn=run_single_suite,
            inputs=[suite, model, output, seed, limit, conditions, pack, profile],
            outputs=[html_out, path_out],
        )


# --- Compare tab -----------------------------------------------------------


def _compare_tab() -> None:
    with gr.Tab("Compare"):
        gr.Markdown("Compare two run JSON files — equivalent to `audiobench compare`.")
        with gr.Row():
            a = gr.Textbox(label="run A")
            b = gr.Textbox(label="run B")
        allow = gr.Checkbox(label="allow mismatched prompt (sound-id)", value=False)
        cli_box = gr.Code(language="shell", value="", label="equivalent CLI")
        for c in (a, b, allow):
            c.change(fn=cli_for_compare, inputs=[a, b, allow], outputs=cli_box)
        btn = gr.Button("Compare", variant="primary")
        html_out = gr.HTML("")
        btn.click(fn=compare_runs, inputs=[a, b, allow], outputs=html_out)


# --- Inspect tab -----------------------------------------------------------


def _inspect_tab() -> None:
    with gr.Tab("Inspect"):
        gr.Markdown("Drill into a per-clip (ASR) or per-mixture (sound-id) record — equivalent to `audiobench inspect`.")
        run_path = gr.Textbox(label="run JSON path")
        with gr.Row():
            kind = gr.Radio(choices=["clip", "mixture"], value="clip", label="record kind")
            idx = gr.Number(value=1, precision=0, label="1-based index")
        cli_box = gr.Code(language="shell", value="", label="equivalent CLI")
        for c in (run_path, kind, idx):
            c.change(fn=cli_for_inspect, inputs=[run_path, kind, idx], outputs=cli_box)
        btn = gr.Button("Inspect", variant="primary")
        html_out = gr.HTML("")
        btn.click(fn=inspect_run, inputs=[run_path, kind, idx], outputs=html_out)


# --- Gate tab --------------------------------------------------------------


_DEFAULT_GATE_TEMPLATE = """# Threshold spec — each top-level key matches one suite.
sound_id:
  min_weighted_recall: 0.6
  max_weighted_fpr: 0.1
asr_robust:
  max_weighted_mean_wer: 30.0
"""


def _gate_tab() -> None:
    with gr.Tab("Gate"):
        gr.Markdown("Evaluate a run JSON against thresholds — equivalent to `audiobench gate`.")
        run_path = gr.Textbox(label="run JSON path")
        gate_yaml = gr.Code(language="yaml", value=_DEFAULT_GATE_TEMPLATE, label="thresholds")
        cli_box = gr.Code(language="shell", value="", label="equivalent CLI")
        run_path.change(fn=cli_for_gate, inputs=run_path, outputs=cli_box)
        btn = gr.Button("Evaluate", variant="primary")
        html_out = gr.HTML("")
        json_out = gr.Code(language="json", label="report JSON")
        btn.click(fn=evaluate_gate_path, inputs=[run_path, gate_yaml], outputs=[html_out, json_out])


# --- Push tab --------------------------------------------------------------


def _push_tab() -> None:
    with gr.Tab("Push"):
        gr.Markdown("Publish a run JSON to a Hugging Face dataset — equivalent to `audiobench push`.")
        run_path = gr.Textbox(label="run JSON path")
        with gr.Row():
            repo = gr.Textbox(label="repo (org/name; blank = auto from hf auth login)")
            token = gr.Textbox(label="HF token (optional)", type="password")
        with gr.Row():
            notes = gr.Textbox(label="notes")
            tags = gr.Textbox(label="tags (comma)")
        with gr.Row():
            dry_run = gr.Checkbox(label="dry run", value=True)
            overwrite = gr.Checkbox(label="overwrite existing", value=False)
            private = gr.Checkbox(label="private repo (if created)", value=False)
        cli_box = gr.Code(language="shell", value="", label="equivalent CLI")
        for c in (run_path, repo, dry_run):
            c.change(fn=cli_for_push, inputs=[run_path, repo, dry_run], outputs=cli_box)
        btn = gr.Button("Push", variant="primary")
        html_out = gr.HTML("")
        btn.click(
            fn=push_run,
            inputs=[run_path, repo, token, notes, tags, dry_run, overwrite, private],
            outputs=html_out,
        )
