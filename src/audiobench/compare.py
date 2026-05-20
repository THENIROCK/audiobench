"""Cross-suite ``audiobench compare`` rendering.

Dispatches on the ``suite`` field in each run JSON:
- asr-robust: lower WER wins
- asr-hallucination: lower hallucination wins (+ finding status)
- sound-id: higher recall wins / lower FPR wins
All modes write console tables and produce JSON summaries.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


def _delta_color_lower_better(delta: float) -> str:
    if delta < 0:
        return f"[green]{delta:+.2f}[/green]"
    if delta > 0:
        return f"[red]{delta:+.2f}[/red]"
    return f"[yellow]{delta:+.2f}[/yellow]"


def _delta_color_higher_better(delta: float) -> str:
    if delta > 0:
        return f"[green]{delta:+.2f}[/green]"
    if delta < 0:
        return f"[red]{delta:+.2f}[/red]"
    return f"[yellow]{delta:+.2f}[/yellow]"


def _winner(a_val: float, b_val: float, a_name: str, b_name: str, *, lower_better: bool) -> str:
    if a_val == b_val:
        return "tie"
    if lower_better:
        return a_name if a_val < b_val else b_name
    return a_name if a_val > b_val else b_name


def _format_winner(winner: str, a_name: str, b_name: str) -> str:
    if winner == "tie":
        return "[yellow]tie[/yellow]"
    if winner == b_name:
        return f"[green]{winner}[/green]"
    if winner == a_name:
        return f"[red]{winner}[/red]"
    return winner


def _format_validation_status(status: str) -> str:
    if status == "validated":
        return "[green]validated[/green]"
    if status == "candidate":
        return "[yellow]candidate[/yellow]"
    if status == "rejected":
        return "[red]rejected[/red]"
    return status


def _top_finding(run: dict) -> dict | None:
    top = run.get("top_finding")
    if isinstance(top, dict) and top:
        return top
    findings = run.get("findings") or []
    if findings and isinstance(findings[0], dict):
        return findings[0]
    return None


class CompareMismatchError(ValueError):
    """Raised when two runs cannot be compared because of incompatible config."""


def render_run_pair(
    left: dict,
    right: dict,
    *,
    console: Console | None = None,
    allow_mismatched_prompt: bool = False,
) -> dict[str, Any]:
    left_suite = left.get("suite")
    right_suite = right.get("suite")
    if left_suite != right_suite:
        raise ValueError(
            f"cannot compare runs with different suites: {left_suite!r} vs {right_suite!r}"
        )
    if left_suite == "ab/sound-id":
        return _render_sound_id(
            left,
            right,
            console=console or Console(),
            allow_mismatched_prompt=allow_mismatched_prompt,
        )
    if left_suite == "ab/asr-hallucination":
        return _render_asr_hallucination(left, right, console=console or Console())
    if left_suite in {
        "ab/fidelity-roundtrip",
        "ab/psychoacoustic-masking",
        "ab/phase-coherence",
        "ab/sed-urban",
        "ab/diarization-cw",
    }:
        return _render_signal(left, right, console=console or Console())
    return _render_asr_robust(left, right, console=console or Console())


_SIGNAL_HEADLINES = {
    "ab/fidelity-roundtrip": [
        ("weighted_si_sdr_db", "weighted SI-SDR (dB)", False),
        ("max_true_peak_dbtp", "max true peak (dBTP)", True),
        ("mean_loudness_delta_lu", "mean loudness Δ (LU)", True),
    ],
    "ab/psychoacoustic-masking": [
        ("masking_respect_score", "masking respect score", False),
        ("mean_in_band_snr_delta_db", "mean in-band SNR Δ (dB)", True),
        ("mean_inaudible_energy_delta_db", "mean inaudible energy Δ (dB)", True),
    ],
    "ab/phase-coherence": [
        ("phase_coherence_score", "phase coherence score", False),
        ("mean_polarity_score", "mean polarity score", False),
    ],
    "ab/sed-urban": [
        ("event_f1_iou50", "event-F1 @ IoU 0.5", False),
        ("event_precision_iou50", "event precision", False),
        ("event_recall_iou50", "event recall", False),
        ("segment_f1_1s", "segment-F1 (1s)", False),
    ],
    "ab/diarization-cw": [
        ("der", "DER", True),
        ("miss_rate", "miss rate", True),
        ("false_alarm_rate", "false-alarm rate", True),
        ("confusion_rate", "confusion rate", True),
        ("mean_speaker_count_error", "speaker-count error", True),
    ],
}


def _render_signal(left: dict, right: dict, *, console: Console) -> dict[str, Any]:
    suite = left.get("suite", "")
    a_name = str(left.get("model"))
    b_name = str(right.get("model"))
    headlines = _SIGNAL_HEADLINES.get(suite, [])
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("metric")
    table.add_column(a_name, justify="right")
    table.add_column(b_name, justify="right")
    table.add_column("delta (B - A)", justify="right")
    table.add_column("winner")
    summary: dict[str, Any] = {"suite": suite, "a": a_name, "b": b_name, "metrics": {}}
    left_headline = left.get("headline", {}) or {}
    right_headline = right.get("headline", {}) or {}
    for key, label, lower_better in headlines:
        a_val = left_headline.get(key)
        b_val = right_headline.get(key)
        if a_val is None or b_val is None:
            continue
        a_val_f = float(a_val)
        b_val_f = float(b_val)
        delta = b_val_f - a_val_f
        compare_a = abs(a_val_f) if lower_better else a_val_f
        compare_b = abs(b_val_f) if lower_better else b_val_f
        winner = _winner(compare_a, compare_b, a_name, b_name, lower_better=lower_better)
        if lower_better:
            delta_text = _delta_color_lower_better(delta)
        else:
            delta_text = _delta_color_higher_better(delta)
        table.add_row(
            label,
            f"{a_val_f:.3f}",
            f"{b_val_f:.3f}",
            delta_text,
            _format_winner(winner, a_name, b_name),
        )
        summary["metrics"][key] = {
            "a": a_val_f,
            "b": b_val_f,
            "delta": delta,
            "winner": winner,
        }
    console.print(table)
    return summary


def _check_prompt_compat(left: dict, right: dict) -> None:
    fields = ("prompt_version", "parser_version", "prompt_ensemble")
    for key in fields:
        if left.get(key) != right.get(key):
            raise CompareMismatchError(
                f"runs disagree on {key}: A={left.get(key)!r} vs B={right.get(key)!r}. "
                "Re-run with matching prompts, or pass --allow-mismatched-prompt."
            )


def _render_asr_robust(left: dict, right: dict, *, console: Console) -> dict[str, Any]:
    conditions: list[str] = []
    seen: set[str] = set()
    for name in left.get("conditions", []) + right.get("conditions", []):
        if name not in seen:
            seen.add(name)
            conditions.append(name)

    deltas: dict[str, float] = {}
    winners: dict[str, str] = {}
    a_name = str(left.get("model"))
    b_name = str(right.get("model"))

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("condition")
    table.add_column(f"{a_name} WER", justify="right")
    table.add_column(f"{b_name} WER", justify="right")
    table.add_column("Δ (b-a)", justify="right")
    table.add_column("winner", justify="left")

    for condition in conditions:
        a_val = left["per_condition_wer"].get(condition)
        b_val = right["per_condition_wer"].get(condition)
        if a_val is None or b_val is None:
            continue
        delta = float(b_val) - float(a_val)
        deltas[condition] = delta
        winner = _winner(float(a_val), float(b_val), a_name, b_name, lower_better=True)
        winners[condition] = winner
        table.add_row(
            condition,
            f"{a_val:.2f}",
            f"{b_val:.2f}",
            _delta_color_lower_better(delta),
            _format_winner(winner, a_name, b_name),
        )

    weighted_mean_delta = float(right.get("weighted_mean_wer", 0.0)) - float(
        left.get("weighted_mean_wer", 0.0)
    )
    mean_winner = _winner(
        float(left["weighted_mean_wer"]),
        float(right["weighted_mean_wer"]),
        a_name,
        b_name,
        lower_better=True,
    )
    table.add_row(
        "weighted mean",
        f"{left['weighted_mean_wer']:.2f}",
        f"{right['weighted_mean_wer']:.2f}",
        _delta_color_lower_better(weighted_mean_delta),
        _format_winner(mean_winner, a_name, b_name),
    )
    console.print(table)

    return {
        "suite": left.get("suite"),
        "model_a": a_name,
        "model_b": b_name,
        "weighted_mean_delta": weighted_mean_delta,
        "per_condition_delta": deltas,
        "per_condition_winner": winners,
    }


def _render_asr_hallucination(left: dict, right: dict, *, console: Console) -> dict[str, Any]:
    conditions: list[str] = []
    seen: set[str] = set()
    for name in left.get("conditions", []) + right.get("conditions", []):
        if name not in seen:
            seen.add(name)
            conditions.append(name)

    deltas: dict[str, float] = {}
    winners: dict[str, str] = {}
    a_name = str(left.get("model"))
    b_name = str(right.get("model"))

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("domain")
    table.add_column(f"{a_name} hallucination", justify="right")
    table.add_column(f"{b_name} hallucination", justify="right")
    table.add_column("Δ (b-a)", justify="right")
    table.add_column("winner", justify="left")

    for condition in conditions:
        a_val = (left.get("per_condition_metrics", {}).get(condition) or {}).get("non_speech_hallucination_rate")
        b_val = (right.get("per_condition_metrics", {}).get(condition) or {}).get("non_speech_hallucination_rate")
        if a_val is None or b_val is None:
            continue
        delta = float(b_val) - float(a_val)
        deltas[condition] = delta
        winner = _winner(float(a_val), float(b_val), a_name, b_name, lower_better=True)
        winners[condition] = winner
        table.add_row(
            condition,
            f"{float(a_val):.2f}",
            f"{float(b_val):.2f}",
            _delta_color_lower_better(delta),
            _format_winner(winner, a_name, b_name),
        )

    weighted_delta = float(right.get("weighted_hallucination_rate", 0.0)) - float(
        left.get("weighted_hallucination_rate", 0.0)
    )
    mean_winner = _winner(
        float(left.get("weighted_hallucination_rate", 0.0)),
        float(right.get("weighted_hallucination_rate", 0.0)),
        a_name,
        b_name,
        lower_better=True,
    )
    table.add_row(
        "weighted",
        f"{float(left.get('weighted_hallucination_rate', 0.0)):.2f}",
        f"{float(right.get('weighted_hallucination_rate', 0.0)):.2f}",
        _delta_color_lower_better(weighted_delta),
        _format_winner(mean_winner, a_name, b_name),
    )
    console.print(table)
    top_left = _top_finding(left)
    top_right = _top_finding(right)
    if top_left or top_right:
        findings_table = Table(show_header=True, header_style="bold", box=None)
        findings_table.add_column("run")
        findings_table.add_column("top finding")
        findings_table.add_column("effect Δ", justify="right")
        findings_table.add_column("q", justify="right")
        findings_table.add_column("status")
        if top_left:
            findings_table.add_row(
                f"A ({a_name})",
                str(top_left.get("title", top_left.get("id", "—"))),
                f"{float(top_left.get('effect_size', 0.0)):+.3f}",
                f"{float(top_left.get('adjusted_p_value', 1.0)):.3f}",
                _format_validation_status(str(top_left.get("status", "unknown"))),
            )
        if top_right:
            findings_table.add_row(
                f"B ({b_name})",
                str(top_right.get("title", top_right.get("id", "—"))),
                f"{float(top_right.get('effect_size', 0.0)):+.3f}",
                f"{float(top_right.get('adjusted_p_value', 1.0)):.3f}",
                _format_validation_status(str(top_right.get("status", "unknown"))),
            )
        console.print()
        console.print(findings_table)
    left_status = (top_left or {}).get("status")
    right_status = (top_right or {}).get("status")
    return {
        "suite": left.get("suite"),
        "model_a": a_name,
        "model_b": b_name,
        "weighted_hallucination_delta": weighted_delta,
        "per_condition_delta": deltas,
        "per_condition_winner": winners,
        "top_finding_a": top_left,
        "top_finding_b": top_right,
        "finding_status_a": left_status,
        "finding_status_b": right_status,
    }


def _render_sound_id(
    left: dict,
    right: dict,
    *,
    console: Console,
    allow_mismatched_prompt: bool = False,
) -> dict[str, Any]:
    a_name = str(left.get("model"))
    b_name = str(right.get("model"))

    if not allow_mismatched_prompt:
        _check_prompt_compat(left, right)

    pack_ids: list[str] = []
    seen: set[str] = set()
    for pid in list(left.get("pack_summaries", {}).keys()) + list(right.get("pack_summaries", {}).keys()):
        if pid not in seen:
            seen.add(pid)
            pack_ids.append(pid)

    console.print(f"{left.get('suite')} compare")
    console.print(f"  A: {a_name}")
    console.print(f"  B: {b_name}")
    if left.get("seed") == right.get("seed"):
        console.print(f"  seed={left.get('seed')}  identical seed [green]✓[/green]")
    else:
        console.print(f"  seeds: A={left.get('seed')} B={right.get('seed')} [yellow](differ)[/yellow]")
    a_pv = left.get("prompt_version")
    b_pv = right.get("prompt_version")
    a_ensemble = left.get("prompt_ensemble")
    b_ensemble = right.get("prompt_ensemble")
    if a_pv == b_pv and a_ensemble == b_ensemble:
        ens_text = f"ensemble={a_ensemble}" if a_ensemble else "ensemble=off"
        console.print(f"  prompts: version={a_pv}, {ens_text} [green]✓[/green]")
    else:
        console.print(
            f"  prompts: A=v{a_pv}/ens={a_ensemble} vs B=v{b_pv}/ens={b_ensemble} "
            "[yellow](mismatched, --allow-mismatched-prompt)[/yellow]"
        )
    console.print()

    deltas: dict[str, dict[str, float]] = {}
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("pack / condition")
    table.add_column(f"{a_name} recall", justify="right")
    table.add_column(f"{b_name} recall", justify="right")
    table.add_column("Δ (b-a)", justify="right")
    table.add_column("winner", justify="left")

    for pid in pack_ids:
        a_pack = left.get("pack_summaries", {}).get(pid, {})
        b_pack = right.get("pack_summaries", {}).get(pid, {})
        conditions: list[str] = []
        condition_seen: set[str] = set()
        for condition in list(a_pack.get("per_condition", {}).keys()) + list(
            b_pack.get("per_condition", {}).keys()
        ):
            if condition not in condition_seen:
                condition_seen.add(condition)
                conditions.append(condition)
        deltas[pid] = {}
        for condition in conditions:
            a_metrics = a_pack.get("per_condition", {}).get(condition)
            b_metrics = b_pack.get("per_condition", {}).get(condition)
            if not a_metrics or not b_metrics:
                continue
            a_recall = float(a_metrics.get("recall", 0.0))
            b_recall = float(b_metrics.get("recall", 0.0))
            delta = b_recall - a_recall
            deltas[pid][condition] = delta
            winner = _winner(a_recall, b_recall, a_name, b_name, lower_better=False)
            table.add_row(
                f"{pid} / {condition}",
                f"{a_recall:.2f}",
                f"{b_recall:.2f}",
                _delta_color_higher_better(delta),
                _format_winner(winner, a_name, b_name),
            )
    console.print(table)

    headline_table = Table(show_header=True, header_style="bold", box=None)
    headline_table.add_column("metric")
    headline_table.add_column(a_name, justify="right")
    headline_table.add_column(b_name, justify="right")
    headline_table.add_column("Δ", justify="right")

    a_headline = left.get("headline", {})
    b_headline = right.get("headline", {})

    a_understood = int(a_headline.get("components_understood", 0))
    a_present = int(a_headline.get("components_present", 0))
    b_understood = int(b_headline.get("components_understood", 0))
    b_present = int(b_headline.get("components_present", 0))
    headline_table.add_row(
        "components understood",
        f"{a_understood}/{a_present}",
        f"{b_understood}/{b_present}",
        _delta_color_higher_better(float(b_understood - a_understood)),
    )
    headline_table.add_row(
        "weighted recall",
        f"{a_headline.get('weighted_recall', 0.0):.2f}",
        f"{b_headline.get('weighted_recall', 0.0):.2f}",
        _delta_color_higher_better(
            float(b_headline.get("weighted_recall", 0.0) - a_headline.get("weighted_recall", 0.0))
        ),
    )
    headline_table.add_row(
        "weighted FPR",
        f"{a_headline.get('weighted_fpr', 0.0):.2f}",
        f"{b_headline.get('weighted_fpr', 0.0):.2f}",
        _delta_color_lower_better(
            float(b_headline.get("weighted_fpr", 0.0) - a_headline.get("weighted_fpr", 0.0))
        ),
    )

    a_cliff = a_headline.get("solo_quad_cliff")
    b_cliff = b_headline.get("solo_quad_cliff")
    if a_cliff is not None and b_cliff is not None:
        headline_table.add_row(
            "solo→quad cliff",
            f"{a_cliff:+.2f}",
            f"{b_cliff:+.2f}",
            _delta_color_higher_better(float(b_cliff - a_cliff)),
        )
    console.print(headline_table)

    class_deltas = _per_class_deltas(left, right)
    if class_deltas:
        gains = sorted(class_deltas.items(), key=lambda kv: -kv[1])[:3]
        regressions = [item for item in class_deltas.items() if item[1] < 0]
        regressions.sort(key=lambda kv: kv[1])

        if gains and gains[0][1] > 0:
            console.print()
            console.print(f"biggest gains for {b_name} (top 3 classes):")
            for label, delta in gains:
                console.print(f"  {label:20} {delta:+.2f}")
        if regressions:
            console.print(f"regressions for {b_name}:")
            for label, delta in regressions:
                console.print(f"  {label:20} {delta:+.2f}")

    return {
        "suite": left.get("suite"),
        "model_a": a_name,
        "model_b": b_name,
        "per_pack_per_condition_delta": deltas,
        "headline_delta": {
            "components_understood": int(b_understood - a_understood),
            "weighted_recall": float(
                b_headline.get("weighted_recall", 0.0) - a_headline.get("weighted_recall", 0.0)
            ),
            "weighted_fpr": float(
                b_headline.get("weighted_fpr", 0.0) - a_headline.get("weighted_fpr", 0.0)
            ),
        },
        "per_class_delta": class_deltas,
    }


def _per_class_deltas(left: dict, right: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid, summary in left.get("pack_summaries", {}).items():
        a_classes = summary.get("per_class", {})
        b_classes = right.get("pack_summaries", {}).get(pid, {}).get("per_class", {})
        for label in set(a_classes) | set(b_classes):
            a_recall = float(a_classes.get(label, {}).get("recall", 0.0)) if a_classes.get(label) else 0.0
            b_recall = float(b_classes.get(label, {}).get("recall", 0.0)) if b_classes.get(label) else 0.0
            out[label] = b_recall - a_recall
    return out
