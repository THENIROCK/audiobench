from __future__ import annotations

from rich.console import Console
from rich.table import Table


def _format_validation_status(status: str) -> str:
    if status == "validated":
        return "[green]validated[/green]"
    if status == "candidate":
        return "[yellow]candidate[/yellow]"
    if status == "rejected":
        return "[red]rejected[/red]"
    return status


def render_run_summary(result: dict, *, console: Console | None = None) -> None:
    suite = result.get("suite")
    if suite == "ab/sound-id":
        render_sound_id_summary(result, console=console)
        return
    if suite == "ab/asr-hallucination":
        render_asr_hallucination_summary(result, console=console)
        return
    if suite == "ab/fidelity-roundtrip":
        render_fidelity_summary(result, console=console)
        return
    if suite == "ab/psychoacoustic-masking":
        render_psycho_summary(result, console=console)
        return
    if suite == "ab/phase-coherence":
        render_phase_summary(result, console=console)
        return
    if suite == "ab/sed-urban":
        render_sed_summary(result, console=console)
        return
    if suite == "ab/diarization-cw":
        render_diarization_summary(result, console=console)
        return
    render_asr_robust_summary(result, console=console)


def render_sed_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    headline = result.get("headline", {})
    console.print(
        f"{result['suite']} · {result['model']} · "
        f"{result['clip_count']} clips · sr={result['sample_rate']} Hz · seed={result['seed']}"
    )
    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("clip")
    table.add_column("ref", justify="right")
    table.add_column("hyp", justify="right")
    table.add_column("event-F1", justify="right")
    table.add_column("seg-F1", justify="right")
    for entry in result.get("per_clip", []):
        ev = entry.get("event_metrics") or {}
        seg = entry.get("segment_metrics") or {}
        table.add_row(
            entry["clip_id"],
            str(len(entry.get("reference_events") or [])),
            str(len(entry.get("hypothesis_events") or [])),
            f"{ev.get('f1', 0.0):.2f}",
            f"{seg.get('f1', 0.0):.2f}",
        )
    console.print(table)
    console.print(
        f"event-F1@IoU{int(headline.get('iou_threshold', 0.5) * 100)}: "
        f"{headline.get('event_f1_iou50', 0.0):.2f} · "
        f"segment-F1@{headline.get('segment_s', 1.0):.1f}s: "
        f"{headline.get('segment_f1_1s', 0.0):.2f}"
    )
    console.print(_run_tag(result))


def render_diarization_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    headline = result.get("headline", {})
    console.print(
        f"{result['suite']} · {result['model']} · "
        f"{result['clip_count']} clips · sr={result['sample_rate']} Hz · seed={result['seed']}"
    )
    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("clip")
    table.add_column("dur", justify="right")
    table.add_column("ref spk", justify="right")
    table.add_column("hyp spk", justify="right")
    table.add_column("DER", justify="right")
    table.add_column("miss", justify="right")
    table.add_column("FA", justify="right")
    table.add_column("conf", justify="right")
    for entry in result.get("per_clip", []):
        m = entry.get("metrics") or {}
        table.add_row(
            entry["clip_id"],
            f"{entry.get('duration_s', 0.0):.1f}s",
            str(m.get("speaker_count_reference", 0)),
            str(m.get("speaker_count_hypothesis", 0)),
            f"{m.get('der', 0.0):.2f}",
            f"{m.get('miss_rate', 0.0):.2f}",
            f"{m.get('false_alarm_rate', 0.0):.2f}",
            f"{m.get('confusion_rate', 0.0):.2f}",
        )
    console.print(table)
    console.print(
        f"DER: {headline.get('der', 0.0):.3f}  "
        f"(miss={headline.get('miss_rate', 0.0):.3f}, "
        f"FA={headline.get('false_alarm_rate', 0.0):.3f}, "
        f"conf={headline.get('confusion_rate', 0.0):.3f})  "
        f"speaker-count err: {headline.get('mean_speaker_count_error', 0.0):.2f}"
    )
    console.print(_run_tag(result))


def render_fidelity_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    headline = result.get("headline", {})
    console.print(
        f"{result['suite']} · {result['model']} · "
        f"{result['stimulus_count']} stimuli × {len(result['conditions'])} conditions · "
        f"sr={result['sample_rate']} Hz · seed={result['seed']}"
    )
    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("condition")
    table.add_column("mean SI-SDR dB", justify="right")
    table.add_column("MR-STFT L1", justify="right")
    table.add_column("max TP dBTP", justify="right")
    table.add_column("mean LUFS Δ", justify="right")
    for name, metrics in (result.get("per_condition_metrics") or {}).items():
        table.add_row(
            name,
            f"{metrics.get('mean_si_sdr_db', 0.0):.2f}",
            f"{metrics.get('mean_mr_stft_log_l1', 0.0):.3f}",
            f"{metrics.get('max_true_peak_dbtp', 0.0):.2f}",
            f"{metrics.get('mean_loudness_delta_lu', 0.0):+.2f}",
        )
    console.print(table)
    console.print(
        f"weighted SI-SDR: {headline.get('weighted_si_sdr_db', 0.0):.2f} dB · "
        f"max true peak: {headline.get('max_true_peak_dbtp', 0.0):+.2f} dBTP · "
        f"mean loudness Δ: {headline.get('mean_loudness_delta_lu', 0.0):+.2f} LU"
    )
    console.print(_run_tag(result))


def render_psycho_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    headline = result.get("headline", {})
    console.print(
        f"{result['suite']} · {result['model']} · "
        f"{result['stimulus_count']} stimuli · sr={result['sample_rate']} Hz · seed={result['seed']}"
    )
    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("stimulus")
    table.add_column("target Hz", justify="right")
    table.add_column("expected")
    table.add_column("in-band SNR dB", justify="right")
    table.add_column("respected")
    for entry in result.get("per_stimulus", []):
        expected = "audible" if entry["expected_audible"] else "masked"
        respected = "[green]yes[/green]" if entry["respected"] else "[red]no[/red]"
        table.add_row(
            entry["stimulus_id"],
            f"{entry['target_hz']:.0f}",
            expected,
            f"{entry['output_band_snr_db']:.1f}",
            respected,
        )
    console.print(table)
    console.print(
        f"masking respect: {headline.get('masking_respect_score', 0.0):.2f}  "
        f"({headline.get('respected_count', 0)}/{headline.get('stimulus_count', 0)})"
    )
    console.print(_run_tag(result))


def render_phase_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    headline = result.get("headline", {})
    console.print(
        f"{result['suite']} · {result['model']} · "
        f"{result['stimulus_count']} stimuli · sr={result['sample_rate']} Hz · seed={result['seed']}"
    )
    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("stimulus")
    table.add_column("check")
    table.add_column("metric")
    table.add_column("passed")
    for entry in result.get("per_stimulus", []):
        metric_text = ", ".join(f"{k}={v:.3f}" for k, v in (entry.get("metrics") or {}).items())
        passed = "[green]yes[/green]" if entry["passed"] else "[red]no[/red]"
        table.add_row(entry["stimulus_id"], entry["check"], metric_text, passed)
    console.print(table)
    console.print(
        f"phase coherence: {headline.get('phase_coherence_score', 0.0):.2f}  "
        f"({headline.get('passed_count', 0)}/{headline.get('stimulus_count', 0)})  "
        f"mean polarity: {headline.get('mean_polarity_score', 0.0):.2f}"
    )
    console.print(_run_tag(result))


def _run_tag(result: dict) -> str:
    digest = result.get("run_hash", "")
    revision = result.get("revision", "")
    suite = result.get("suite", "").replace("/", "-")
    short = f"{digest[:8]}…{digest[-4:]}" if digest else "no-hash"
    return f"run hash: {suite}@{revision} · {short}"


def render_asr_robust_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    header = (
        f"{result['suite']} · {result['model']} · "
        f"{result['clip_count']} clips × {len(result['conditions'])} conditions · "
        f"seed={result['seed']}"
    )
    console.print(header)
    console.print()

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("condition")
    table.add_column("WER", justify="right")
    table.add_column("Δ vs clean", justify="right")

    baseline_key = "clean" if "clean" in result["per_condition_wer"] else result["conditions"][0]
    clean = result["per_condition_wer"][baseline_key]
    for condition in result["conditions"]:
        value = result["per_condition_wer"][condition]
        delta = "—" if condition == baseline_key else f"{value - clean:+.2f}"
        table.add_row(condition, f"{value:.2f}", delta)

    table.add_row("weighted mean", f"{result['weighted_mean_wer']:.2f}", "")
    console.print(table)
    console.print(
        f"run hash: {result['suite'].replace('/', '-')}@{result['revision']} · "
        f"{result['run_hash'][:8]}…{result['run_hash'][-4:]}"
    )


def render_asr_hallucination_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    header = (
        f"{result['suite']} · {result['model']} · "
        f"{result['clip_count']} clips × {len(result['conditions'])} domains · "
        f"seed={result['seed']}"
    )
    console.print(header)
    console.print()

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("domain")
    table.add_column("hallucination", justify="right")
    table.add_column("empty", justify="right")
    table.add_column("insertions", justify="right")
    table.add_column("latency ms", justify="right")

    per_condition = result.get("per_condition_metrics", {})
    runtime = result.get("per_condition_runtime", {})
    for condition in result.get("conditions", []):
        metrics = per_condition.get(condition, {})
        condition_runtime = runtime.get(condition, {})
        latency = condition_runtime.get("mean_latency_ms")
        latency_text = "—" if latency is None else f"{latency:.1f}"
        table.add_row(
            condition,
            f"{float(metrics.get('non_speech_hallucination_rate', 0.0)):.2f}",
            f"{float(metrics.get('non_speech_empty_rate', 0.0)):.2f}",
            f"{float(metrics.get('non_speech_mean_inserted_tokens', 0.0)):.2f}",
            latency_text,
        )

    headline = result.get("headline", {})
    table.add_row(
        "weighted",
        f"{float(result.get('weighted_hallucination_rate', 0.0)):.2f}",
        f"{float(headline.get('non_speech_empty_rate', 0.0)):.2f}",
        f"{float(headline.get('non_speech_mean_inserted_tokens', 0.0)):.2f}",
        "",
    )
    console.print(table)
    findings = result.get("findings") or []
    if findings:
        findings_table = Table(show_header=True, header_style="bold", box=None)
        findings_table.add_column("rank", justify="right")
        findings_table.add_column("finding")
        findings_table.add_column("effect Δ", justify="right")
        findings_table.add_column("95% CI", justify="right")
        findings_table.add_column("q", justify="right")
        findings_table.add_column("status")
        for finding in findings[:5]:
            findings_table.add_row(
                str(int(finding.get("rank", 0))),
                str(finding.get("title", finding.get("id", "finding"))),
                f"{float(finding.get('effect_size', 0.0)):+.3f}",
                f"[{float(finding.get('ci_lower', 0.0)):+.3f}, {float(finding.get('ci_upper', 0.0)):+.3f}]",
                f"{float(finding.get('adjusted_p_value', 1.0)):.3f}",
                _format_validation_status(str(finding.get("status", "unknown"))),
            )
        console.print()
        console.print(findings_table)
    validation = result.get("validation_summary") or {}
    status_counts = validation.get("status_counts") or {}
    if status_counts:
        publishable = bool(validation.get("publishable", False))
        badge = "[green]publishable[/green]" if publishable else "[yellow]not publishable[/yellow]"
        console.print(
            f"validation gate: validated={int(status_counts.get('validated', 0))} · "
            f"candidate={int(status_counts.get('candidate', 0))} · "
            f"rejected={int(status_counts.get('rejected', 0))} · {badge}"
        )
    console.print(
        f"run hash: {result['suite'].replace('/', '-')}@{result['revision']} · "
        f"{result['run_hash'][:8]}…{result['run_hash'][-4:]}"
    )


def render_sound_id_summary(result: dict, *, console: Console | None = None) -> None:
    console = console or Console()
    packs = result.get("packs", [])
    profile = result.get("profile")
    profile_text = f" · profile={profile}" if profile else ""
    pack_str = ", ".join(packs) if packs else "(none)"
    header = (
        f"{result['suite']} · {result['model']} · "
        f"packs=({pack_str}) · seed={result['seed']}{profile_text}"
    )
    console.print(header)
    prompt_version = result.get("prompt_version")
    if prompt_version:
        ensemble = result.get("prompt_ensemble")
        ensemble_text = f"ensemble={ensemble}" if ensemble else "ensemble=off"
        console.print(
            f"  prompts: version={prompt_version} · parser={result.get('parser_version', 'v1')} · "
            f"{ensemble_text}"
        )

    skipped = result.get("skipped_packs", []) or []
    for entry in skipped:
        console.print(f"  [yellow]skipped[/yellow] {entry['pack']}: {entry['reason']}")
    console.print()

    pack_summaries: dict[str, dict] = result.get("pack_summaries", {})
    for pack_id, summary in pack_summaries.items():
        license_tag = summary.get("license_tag", "")
        title_suffix = f" ({license_tag})" if license_tag else ""
        table = Table(
            title=f"pack={pack_id}{title_suffix}",
            show_header=True,
            header_style="bold",
            box=None,
            title_justify="left",
        )
        table.add_column("condition")
        table.add_column("recall", justify="right")
        table.add_column("precision", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("FPR", justify="right")

        for condition in ("solo", "pair", "triple", "quad", "custom"):
            metrics = summary.get("per_condition", {}).get(condition)
            if not metrics:
                continue
            table.add_row(
                condition,
                f"{metrics['recall']:.2f}",
                f"{metrics['precision']:.2f}",
                f"{metrics['f1']:.2f}",
                f"{metrics['fpr']:.2f}",
            )
        totals = summary.get("totals", {})
        table.add_row(
            "[bold]all[/bold]",
            f"{totals.get('recall', 0.0):.2f}",
            f"{totals.get('precision', 0.0):.2f}",
            f"{totals.get('f1', 0.0):.2f}",
            f"{totals.get('fpr', 0.0):.2f}",
        )
        console.print(table)

    headline = result.get("headline", {})
    understood = headline.get("components_understood", 0)
    present = headline.get("components_present", 0)
    weighted_recall = headline.get("weighted_recall", 0.0)
    weighted_fpr = headline.get("weighted_fpr", 0.0)
    cliff = headline.get("solo_quad_cliff")
    console.print()
    console.print(
        f"components understood: [bold]{understood} / {present}[/bold]   "
        f"weighted recall: {weighted_recall:.2f}   weighted FPR: {weighted_fpr:.2f}"
    )
    if cliff is not None:
        console.print(f"solo→quad cliff: {cliff:+.2f} (negative = harder under more components)")
    console.print(
        f"run hash: {result['suite'].replace('/', '-')}@{result['revision']} · "
        f"{result['run_hash'][:8]}…{result['run_hash'][-4:]}"
    )
