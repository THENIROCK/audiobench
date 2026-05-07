from __future__ import annotations

from rich.console import Console
from rich.table import Table


def render_run_summary(result: dict, *, console: Console | None = None) -> None:
    suite = result.get("suite")
    if suite == "ab/sound-id":
        render_sound_id_summary(result, console=console)
        return
    render_asr_robust_summary(result, console=console)


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
