from __future__ import annotations

from rich.console import Console
from rich.table import Table


def render_run_summary(result: dict, *, console: Console | None = None) -> None:
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
    console.print(f"run hash: {result['suite'].replace('/', '-')}@{result['revision']} · {result['run_hash'][:8]}…{result['run_hash'][-4:]}")
