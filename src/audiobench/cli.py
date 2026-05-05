from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from audiobench.hashing import sha256_text, stable_json
from audiobench.report import render_run_summary
from audiobench.suites.asr_robust import SUITE_ID, SUITE_REVISION, load_manifest, run_suite


app = typer.Typer(help="audiobench: evaluation suite for audio ML models")
console = Console()


@app.command("list")
def list_suites() -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("suite id")
    table.add_column("domain")
    table.add_column("tasks", justify="right")
    table.add_column("status")
    table.add_row("ab/asr-robust", "speech recognition", "5", "stable (MVP)")
    table.add_row("ab/separation-musdb+", "source separation", "6", "in design")
    table.add_row("ab/tagging-audioset-v2", "audio tagging", "9", "in design")
    table.add_row("ab/diarization-cw", "speaker diarization", "5", "in design")
    table.add_row("ab/music-tag-mtg", "music understanding", "8", "in design")
    table.add_row("ab/sed-urban", "sound event detection", "7", "in design")
    table.add_row("ab/codec-perceptual", "neural codecs", "4", "in design")
    table.add_row("ab/tts-eval", "speech synthesis", "6", "in design")
    console.print(table)


@app.command("info")
def info(suite_id: str) -> None:
    if suite_id != SUITE_ID:
        raise typer.BadParameter(f"only {SUITE_ID} is implemented in this MVP")
    manifest = load_manifest()
    console.print(f"[bold]{SUITE_ID}[/bold] ({SUITE_REVISION})")
    console.print("domain: speech recognition")
    console.print(f"clips: {len(manifest['clips'])}")
    console.print("conditions:")
    for name in manifest["conditions"]:
        console.print(f"  - {name}")


@app.command("run")
def run(
    suite_id: str,
    model: str = typer.Option("tiny", "--model", help="Whisper model name (tiny/base/small/...)"),
    output: Path | None = typer.Option(None, "--output", help="Path to write run JSON"),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed"),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of clips for quick checks"),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    if suite_id != SUITE_ID:
        raise typer.BadParameter(f"only {SUITE_ID} is implemented in this MVP")

    normalized_model = model.replace("whisper-", "")
    result = run_suite(model_name=normalized_model, seed=seed, limit=limit)
    if output is None:
        output = Path("results") / f"run-{result['run_hash'][:8]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if json_out:
        console.print(json.dumps(result))
    else:
        render_run_summary(result, console=console)
        console.print(f"wrote: {output}")


@app.command("push")
def push(run_file: Path) -> None:
    if not run_file.exists():
        raise typer.BadParameter(f"file not found: {run_file}")
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    signed_payload = {
        "suite": payload["suite"],
        "revision": payload["revision"],
        "run_hash": payload["run_hash"],
        "payload_sha256": sha256_text(stable_json(payload)),
        "mode": "stub-no-network",
    }
    console.print(json.dumps(signed_payload, indent=2))
    console.print("push is a local stub in MVP mode; no data was sent.")


if __name__ == "__main__":
    app()
