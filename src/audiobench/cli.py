from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from audiobench.compare import CompareMismatchError, render_run_pair
from audiobench.gating import (
    GateConfigError,
    GateReport,
    evaluate_gate,
    format_console as format_gate_console,
    load_thresholds,
    merge_inline_thresholds,
)
from audiobench.hashing import sha256_text
from audiobench.junit import (
    JUnitCase,
    JUnitSuite,
    gate_report_to_junit,
    render_junit_xml,
)
from audiobench.matrix import (
    CellResult,
    MatrixCell,
    MatrixConfigError,
    MatrixPlan,
    build_cells_from_cli,
    build_summary,
    load_matrix_file,
    run_matrix as run_matrix_cells,
)
from audiobench.leaderboard import (
    DEFAULT_DATASET_ENV,
    DEFAULT_SPACE_ENV,
    build_submission_record,
    default_dataset_repo_id,
    push_submission,
)
from audiobench.mixing import MixSource, mix_sources
from audiobench.models.asr_registry import list_models as list_asr_models
from audiobench.models.diarization_registry import list_models as list_diarization_models
from audiobench.models.registry import list_models as list_sound_id_models
from audiobench.models.sed_registry import list_models as list_sed_models
from audiobench.models.signal_registry import list_models as list_signal_models
from audiobench.models.whisper import warmup_model
from audiobench.packs import (
    UserCacheResolver,
    filter_to_available,
    list_pack_ids,
    load_pack_manifest,
    make_resolver,
)
from audiobench.prompts import (
    PromptFormatError,
    bundled_prompts_text,
    export_default_prompts,
    load_prompts,
)
from audiobench.recipes import MixtureSpec, load_recipes, parse_inline_mix
from audiobench.report import render_run_summary
from audiobench.suites import (
    asr_suite_ids,
    list_suite_specs,
    signal_suite_ids,
    sound_id as sound_id_suite,
    temporal_suite_ids,
)
from audiobench.suites.asr_hallucination import (
    SUITE_ID as ASR_HALLUCINATION_SUITE_ID,
    SUITE_REVISION as ASR_HALLUCINATION_SUITE_REVISION,
    load_manifest as load_asr_hallucination_manifest,
    run_suite as run_asr_hallucination_suite,
)
from audiobench.suites.asr_robust import (
    SUITE_ID as ASR_SUITE_ID,
    SUITE_REVISION as ASR_SUITE_REVISION,
    load_manifest as load_asr_manifest,
    run_suite as run_asr_suite,
)
from audiobench.suites.fidelity_roundtrip import (
    SUITE_ID as FIDELITY_SUITE_ID,
    SUITE_REVISION as FIDELITY_SUITE_REVISION,
    load_manifest as load_fidelity_manifest,
    run_suite as run_fidelity_suite,
)
from audiobench.suites.phase_coherence import (
    SUITE_ID as PHASE_SUITE_ID,
    SUITE_REVISION as PHASE_SUITE_REVISION,
    load_manifest as load_phase_manifest,
    run_suite as run_phase_suite,
)
from audiobench.suites.psychoacoustic_masking import (
    SUITE_ID as PSYCHO_SUITE_ID,
    SUITE_REVISION as PSYCHO_SUITE_REVISION,
    load_manifest as load_psycho_manifest,
    run_suite as run_psycho_suite,
)
from audiobench.suites.sed_urban import (
    SUITE_ID as SED_SUITE_ID,
    SUITE_REVISION as SED_SUITE_REVISION,
    load_manifest as load_sed_manifest,
    run_suite as run_sed_suite,
)
from audiobench.suites.diarization_cw import (
    SUITE_ID as DIAR_SUITE_ID,
    SUITE_REVISION as DIAR_SUITE_REVISION,
    load_manifest as load_diar_manifest,
    run_suite as run_diar_suite,
)
from audiobench.telemetry import (
    maybe_prompt_for_consent,
    suite_from_run_json,
    track_command,
)


app = typer.Typer(
    help="audiobench: evaluation suite for audio ML models",
    no_args_is_help=False,
)
mix_app = typer.Typer(help="Render mixtures without running probes", no_args_is_help=True)
app.add_typer(mix_app, name="mix")
prompts_app = typer.Typer(help="Inspect and export the ab/sound-id prompt set", no_args_is_help=True)
app.add_typer(prompts_app, name="prompts")
console = Console()


PHONON_LOGO = r"""[bold cyan]
        ___           ___           ___           ___
       /   \         /   \         /   \         /   \
  ____/     \_______/     \_______/     \_______/     \____[/]

         [bold]_[/]
   [bold]_ __ | |__   ___  _ __   ___  _ __[/]
  [bold]| '_ \| '_ \ / _ \| '_ \ / _ \| '_ \ [/]
  [bold]| |_) | | | | (_) | | | | (_) | | | |[/]
  [bold]| .__/|_| |_|\___/|_| |_|\___/|_| |_|[/]
  [bold]|_|[/]

  [dim]o --- o --- o --- o --- o --- o --- o --- o --- o[/]
"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    gui: bool = typer.Option(False, "--gui", help="Launch the local Gradio GUI."),
    gui_port: int = typer.Option(7861, "--gui-port", help="Port for the GUI server."),
    gui_host: str = typer.Option(
        "127.0.0.1", "--gui-host", help="Host/interface to bind the GUI to."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Don't auto-open a browser when launching the GUI."
    ),
) -> None:
    maybe_prompt_for_consent()
    if gui:
        from audiobench.gui import launch as launch_gui

        launch_gui(host=gui_host, port=gui_port, open_browser=not no_open)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(PHONON_LOGO)
        console.print(
            "  [dim]run `audiobench --help` for commands, or `audiobench --gui` for the local GUI[/]"
        )


def _dump_json(data: dict[str, Any], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True)
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def _parse_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _make_run_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _sound_id_progress_callback(progress: Progress, task_id: TaskID):
    task_total: int | None = None

    def handle(event: dict[str, Any]) -> None:
        nonlocal task_total
        kind = event.get("event")
        if kind == "start":
            task_total = event.get("total_mixtures") or None
            progress.update(
                task_id,
                total=task_total,
                description=f"running {event.get('suite', 'ab/sound-id')} · {event.get('model', '')}",
            )
            return
        if kind == "mixture_start":
            progress.update(
                task_id,
                description=(
                    f"{event.get('pack')}:{event.get('condition')} · "
                    f"{event.get('mixture_name')}"
                ),
            )
            return
        if kind == "probe_start":
            suffix = ""
            prompt_total = event.get("prompt_total")
            if prompt_total and prompt_total > 1:
                suffix = f" prompt {event.get('prompt_index')}/{prompt_total}"
            progress.update(
                task_id,
                description=(
                    f"{event.get('pack')}:{event.get('condition')} · "
                    f"{event.get('mixture_name')} · {event.get('label')}{suffix}"
                ),
            )
            return
        if kind == "probe_done":
            answered = "yes" if event.get("answered_yes") else "no"
            expected = "present" if event.get("expected") else "absent"
            prompt_total = event.get("prompt_total")
            prompt_suffix = ""
            if prompt_total and prompt_total > 1:
                prompt_suffix = f" · prompt {event.get('prompt_index')}/{prompt_total}"
            title = (
                f"{event.get('suite', 'ab/sound-id')} · "
                f"{event.get('pack')}:{event.get('condition')} · "
                f"{event.get('mixture_name')}{prompt_suffix}"
            )
            progress.console.print(
                Panel(
                    Group(
                        f"[bold]Label[/bold] {escape(str(event.get('label')))} "
                        f"[dim]({expected})[/dim]",
                        f"[bold]Question[/bold] {escape(str(event.get('prompt', '')))}",
                        f"[bold]LLM response[/bold] {escape(str(event.get('raw_answer', '')).strip() or '(empty)')}",
                        f"[bold]Parsed[/bold] {answered}",
                    ),
                    title=escape(title),
                    border_style="green" if event.get("answered_yes") else "yellow",
                    expand=False,
                )
            )
            return
        if isinstance(kind, str) and kind.startswith("agent_"):
            title = (
                f"{event.get('suite', 'ab/sound-id')} · "
                f"{event.get('pack')}:{event.get('condition')} · "
                f"{event.get('mixture_name')} · {event.get('label')}"
            )
            body = _format_agent_event(event)
            progress.console.print(
                Panel(
                    body,
                    title=escape(title),
                    border_style="blue" if "error" not in body.lower() else "red",
                    expand=False,
                )
            )
            return
        if kind == "mixture_done":
            progress.advance(task_id)
            return
        if kind == "done":
            if task_total is not None:
                progress.update(task_id, completed=task_total, description="finalizing run")
            else:
                progress.update(task_id, description="finalizing run")

    return handle


def _format_agent_event(event: dict[str, Any]) -> str:
    kind = event.get("event")
    if kind == "agent_llm_start":
        return "[bold]Agent[/bold] requesting run_python tool call"
    if kind == "agent_tool_call":
        code = str(event.get("code", "")).strip() or "(empty)"
        return f"[bold]run_python code[/bold]\n{escape(_truncate_display(code, 1600))}"
    if kind == "agent_tool_output":
        output = str(event.get("output", "")).strip() or "(empty)"
        return f"[bold]run_python output[/bold]\n{escape(_truncate_display(output, 1600))}"
    if kind == "agent_final_answer":
        answer = str(event.get("answer", "")).strip() or "(empty)"
        return f"[bold]Final LLM answer[/bold] {escape(answer)}"
    if kind == "agent_direct_answer":
        answer = str(event.get("answer", "")).strip() or "(empty)"
        return f"[bold]Direct LLM answer[/bold] {escape(answer)}"
    return escape(str(event))


def _truncate_display(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "\n... [truncated]"


def _asr_progress_callback(progress: Progress, task_id: TaskID):
    task_total: int | None = None

    def handle(event: dict[str, Any]) -> None:
        nonlocal task_total
        kind = event.get("event")
        if kind == "start":
            task_total = event.get("total_steps") or None
            progress.update(
                task_id,
                total=task_total,
                description=f"running {event.get('suite', 'ab/asr-robust')} · {event.get('model', '')}",
            )
            return
        if kind == "condition_start":
            progress.update(
                task_id,
                description=(
                    f"clip {event.get('clip_index')}/{event.get('clip_total')} · "
                    f"{event.get('condition')}"
                ),
            )
            return
        if kind == "condition_done":
            progress.advance(task_id)
            return
        if kind == "done":
            if task_total is not None:
                progress.update(task_id, completed=task_total, description="finalizing run")
            else:
                progress.update(task_id, description="finalizing run")

    return handle


@app.command("list")
def list_suites() -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("suite id")
    table.add_column("domain")
    table.add_column("tasks", justify="right")
    table.add_column("status")
    for spec in list_suite_specs():
        table.add_row(spec.suite_id, spec.domain, spec.tasks, spec.status)
    console.print(table)


@app.command("list-packs")
def list_packs() -> None:
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("pack")
    table.add_column("title")
    table.add_column("license")
    table.add_column("status")
    statuses = {pid: (ok, reason) for pid, ok, reason in filter_to_available(list_pack_ids())}
    for pid in list_pack_ids():
        manifest = load_pack_manifest(pid)
        ok, reason = statuses.get(pid, (False, "unknown"))
        if ok:
            status = "[green]available[/green]"
        else:
            status = f"[yellow]missing[/yellow] ({reason})"
        table.add_row(pid, manifest.title, manifest.license_tag or manifest.license, status)
    console.print(table)


@app.command("list-models")
def list_models(
    suite_id: str | None = typer.Option(
        None,
        "--suite",
        help="Filter to one suite (e.g. ab/sound-id, ab/asr-robust, ab/fidelity-roundtrip).",
    ),
) -> None:
    model_suite_ids = (
        asr_suite_ids()
        | signal_suite_ids()
        | temporal_suite_ids()
        | {sound_id_suite.SUITE_ID}
    )
    if suite_id and suite_id not in model_suite_ids:
        raise typer.BadParameter(f"unknown suite: {suite_id}")
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("suite")
    table.add_column("model id")
    table.add_column("notes")
    if suite_id in (None, sound_id_suite.SUITE_ID):
        for model_name in list_sound_id_models():
            table.add_row(sound_id_suite.SUITE_ID, model_name, "adapter id")
    if suite_id is None or suite_id in asr_suite_ids():
        for model_name in list_asr_models():
            note = "Whisper checkpoint pattern" if model_name == "whisper-*" else "adapter id"
            if suite_id is None:
                table.add_row(ASR_SUITE_ID, model_name, note)
                table.add_row(ASR_HALLUCINATION_SUITE_ID, model_name, note)
            else:
                table.add_row(suite_id, model_name, note)
    if suite_id is None or suite_id in signal_suite_ids():
        for model_name in list_signal_models():
            if suite_id is None:
                for sig_id in sorted(signal_suite_ids()):
                    table.add_row(sig_id, model_name, "audio processor adapter")
            else:
                table.add_row(suite_id, model_name, "audio processor adapter")
    if suite_id in (None, SED_SUITE_ID):
        for model_name in list_sed_models():
            table.add_row(SED_SUITE_ID, model_name, "SED adapter")
    if suite_id in (None, DIAR_SUITE_ID):
        for model_name in list_diarization_models():
            table.add_row(DIAR_SUITE_ID, model_name, "diarization adapter")
    console.print(table)


@app.command("info")
def info(
    suite_id: str,
    pack: str | None = typer.Option(None, "--pack", help="Inspect a single pack of ab/sound-id"),
) -> None:
    if suite_id == ASR_SUITE_ID:
        manifest = load_asr_manifest()
        console.print(f"[bold]{ASR_SUITE_ID}[/bold] ({ASR_SUITE_REVISION})")
        console.print("domain: speech recognition")
        console.print(f"clips: {len(manifest['clips'])}")
        console.print("conditions:")
        for name in manifest["conditions"]:
            console.print(f"  - {name}")
        console.print(f"models: {', '.join(list_asr_models())}")
        return
    if suite_id == ASR_HALLUCINATION_SUITE_ID:
        manifest = load_asr_hallucination_manifest()
        console.print(f"[bold]{ASR_HALLUCINATION_SUITE_ID}[/bold] ({ASR_HALLUCINATION_SUITE_REVISION})")
        console.print("domain: non-speech ASR hallucination")
        console.print(f"clips: {len(manifest['clips'])}")
        console.print("conditions (slice key=domain):")
        for name in manifest["conditions"]:
            console.print(f"  - {name}")
        console.print("metadata slices: duration_s, domain, noise_type, snr_db, language, accent")
        console.print(f"models: {', '.join(list_asr_models())}")
        return
    if suite_id == FIDELITY_SUITE_ID:
        manifest = load_fidelity_manifest()
        console.print(f"[bold]{FIDELITY_SUITE_ID}[/bold] ({FIDELITY_SUITE_REVISION})")
        console.print("domain: audio fidelity (signal-level)")
        console.print(f"sample rate: {manifest['sample_rate']} Hz")
        console.print(f"stimuli: {len(manifest['stimuli'])}")
        console.print("conditions:")
        for name in manifest["conditions"]:
            console.print(f"  - {name}")
        console.print(f"models: {', '.join(list_signal_models())}")
        return
    if suite_id == PSYCHO_SUITE_ID:
        manifest = load_psycho_manifest()
        console.print(f"[bold]{PSYCHO_SUITE_ID}[/bold] ({PSYCHO_SUITE_REVISION})")
        console.print("domain: psychoacoustics (signal-level)")
        console.print(f"sample rate: {manifest['sample_rate']} Hz")
        console.print(f"stimuli: {len(manifest['stimuli'])}")
        for stim in manifest["stimuli"]:
            tag = "audible" if stim["expected_audible"] else "masked"
            console.print(
                f"  - {stim['id']:30} target={stim['target_hz']:.0f} Hz ({tag})"
            )
        console.print(f"models: {', '.join(list_signal_models())}")
        return
    if suite_id == SED_SUITE_ID:
        manifest = load_sed_manifest()
        console.print(f"[bold]{SED_SUITE_ID}[/bold] ({SED_SUITE_REVISION})")
        console.print("domain: sound event detection (temporal)")
        console.print(f"sample rate: {manifest['sample_rate']} Hz · clip {manifest['clip_duration_s']:.1f}s")
        console.print(f"labels: {', '.join(manifest['labels'])}")
        console.print(
            f"iou threshold: {manifest['iou_threshold']} · segment: {manifest['segment_s']}s"
        )
        console.print(f"clips: {len(manifest['clips'])}")
        for clip in manifest["clips"]:
            console.print(f"  - {clip['clip_id']:18} events={len(clip['events'])}")
        console.print(f"models: {', '.join(list_sed_models())}")
        return
    if suite_id == DIAR_SUITE_ID:
        manifest = load_diar_manifest()
        console.print(f"[bold]{DIAR_SUITE_ID}[/bold] ({DIAR_SUITE_REVISION})")
        console.print("domain: speaker diarization (temporal)")
        console.print(
            f"sample rate: {manifest['sample_rate']} Hz · frame {manifest['frame_s']}s · collar {manifest['collar_s']}s"
        )
        console.print(f"clips: {len(manifest['clips'])}")
        for clip in manifest["clips"]:
            speakers = sorted({turn["speaker_id"] for turn in clip["turns"]})
            console.print(
                f"  - {clip['clip_id']:18} dur={clip['duration_s']:.1f}s "
                f"speakers={len(speakers)} turns={len(clip['turns'])}"
            )
        console.print(f"models: {', '.join(list_diarization_models())}")
        return
    if suite_id == PHASE_SUITE_ID:
        manifest = load_phase_manifest()
        console.print(f"[bold]{PHASE_SUITE_ID}[/bold] ({PHASE_SUITE_REVISION})")
        console.print("domain: phase / multichannel coherence (signal-level)")
        console.print(f"sample rate: {manifest['sample_rate']} Hz")
        console.print(f"stimuli: {len(manifest['stimuli'])}")
        for stim in manifest["stimuli"]:
            console.print(f"  - {stim['id']:30} check={stim['check']}")
        console.print(f"models: {', '.join(list_signal_models())}")
        return
    if suite_id == sound_id_suite.SUITE_ID:
        if pack:
            pack_manifest = load_pack_manifest(pack)
            console.print(f"[bold]{sound_id_suite.SUITE_ID}[/bold] ({sound_id_suite.SUITE_REVISION}) · pack={pack_manifest.id}")
            console.print(f"title: {pack_manifest.title}")
            console.print(f"source: {pack_manifest.source}")
            console.print(f"license: {pack_manifest.license} [{pack_manifest.license_tag}]")
            if pack_manifest.scope_note:
                console.print(f"scope: {pack_manifest.scope_note}")
            console.print(f"labels ({len(pack_manifest.labels)}): {', '.join(pack_manifest.labels)}")
            console.print(f"distractors per mixture: {pack_manifest.distractor_count}")
            console.print(f"mixture counts: {pack_manifest.mixture_counts}")
            if pack_manifest.expected_layout:
                console.print(f"expected on-disk layout: {pack_manifest.expected_layout}")
            resolver = make_resolver(pack_manifest)
            if isinstance(resolver, UserCacheResolver):
                ok = resolver.is_available()
                console.print(f"cache: {resolver.directory}  [{'green' if ok else 'yellow'}]" + ("available" if ok else "missing") + "[/]")
            else:
                console.print("cache: bundled (no download)")
            return
        console.print(f"[bold]{sound_id_suite.SUITE_ID}[/bold] ({sound_id_suite.SUITE_REVISION})")
        console.print("domain: sound event identification on labeled mixtures")
        console.print("conditions: solo, pair, triple, quad (+ custom for --mix/--recipes)")
        console.print("packs: see `audiobench list-packs`")
        console.print(f"models: {', '.join(list_sound_id_models())}")
        return
    raise typer.BadParameter(f"unknown suite: {suite_id}")


@app.command("run")
def run(
    suite_id: str,
    model: str = typer.Option(
        "tiny",
        "--model",
        help="Model adapter id (ab/sound-id) or ASR adapter / whisper checkpoint (ab/asr-*)",
    ),
    output: Path | None = typer.Option(None, "--output", help="Path to write run JSON"),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed"),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of clips/mixtures"),
    conditions: str | None = typer.Option(
        None, "--conditions", help="Comma-separated conditions"
    ),
    pack: str | None = typer.Option(
        None, "--pack", help="Comma-separated pack ids for ab/sound-id"
    ),
    profile: str | None = typer.Option(None, "--profile", help="Run profile (e.g. demo-fast)"),
    mix: list[str] = typer.Option(
        [],
        "--mix",
        help="Inline ad-hoc mixture: '+'-separated labels (repeatable)",
    ),
    recipes: Path | None = typer.Option(
        None, "--recipes", help="YAML/JSON file of mixture recipes"
    ),
    prompts_file: Path | None = typer.Option(
        None,
        "--prompts",
        help="Override prompts with a YAML/JSON file (defaults to bundled prompts.yaml for ab/sound-id)",
    ),
    prompt_ensemble: int | None = typer.Option(
        None,
        "--prompt-ensemble",
        min=1,
        help="Ask N paraphrased prompts per probe and majority-vote (ab/sound-id only)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    pretty_json: bool = typer.Option(False, "--pretty-json", help="Pretty-print JSON output"),
) -> None:
    with track_command("run", suite=suite_id, adapter=model):
        _run_body(
            suite_id=suite_id,
            model=model,
            output=output,
            seed=seed,
            limit=limit,
            conditions=conditions,
            pack=pack,
            profile=profile,
            mix=mix,
            recipes=recipes,
            prompts_file=prompts_file,
            prompt_ensemble=prompt_ensemble,
            json_out=json_out,
            pretty_json=pretty_json,
        )


def _run_body(
    *,
    suite_id: str,
    model: str,
    output: Path | None,
    seed: int,
    limit: int | None,
    conditions: str | None,
    pack: str | None,
    profile: str | None,
    mix: list[str],
    recipes: Path | None,
    prompts_file: Path | None,
    prompt_ensemble: int | None,
    json_out: bool,
    pretty_json: bool,
) -> None:
    if suite_id in asr_suite_ids():
        if mix or recipes or pack or profile or prompts_file or prompt_ensemble is not None:
            raise typer.BadParameter(
                "--mix/--recipes/--pack/--profile/--prompts/--prompt-ensemble only apply to ab/sound-id"
            )
        normalized_model = model.replace("whisper-", "")
        try:
            if suite_id == ASR_SUITE_ID:
                if json_out:
                    result = run_asr_suite(
                        model_name=normalized_model,
                        seed=seed,
                        limit=limit,
                        condition_names=_parse_csv(conditions),
                    )
                else:
                    progress = _make_run_progress()
                    with progress:
                        task_id = progress.add_task("preparing ab/asr-robust", total=None)
                        result = run_asr_suite(
                            model_name=normalized_model,
                            seed=seed,
                            limit=limit,
                            condition_names=_parse_csv(conditions),
                            progress_callback=_asr_progress_callback(progress, task_id),
                        )
            else:
                result = run_asr_hallucination_suite(
                    model_name=model,
                    seed=seed,
                    limit=limit,
                    condition_names=_parse_csv(conditions),
                )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif suite_id in temporal_suite_ids():
        if mix or recipes or pack or profile or prompts_file or prompt_ensemble is not None:
            raise typer.BadParameter(
                "--mix/--recipes/--pack/--profile/--prompts/--prompt-ensemble only apply to ab/sound-id"
            )
        if conditions:
            raise typer.BadParameter(
                f"--conditions does not apply to {suite_id}"
            )
        try:
            if suite_id == SED_SUITE_ID:
                result = run_sed_suite(model_name=model, seed=seed, limit=limit)
            else:
                result = run_diar_suite(model_name=model, seed=seed, limit=limit)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif suite_id in signal_suite_ids():
        if mix or recipes or pack or profile or prompts_file or prompt_ensemble is not None:
            raise typer.BadParameter(
                "--mix/--recipes/--pack/--profile/--prompts/--prompt-ensemble only apply to ab/sound-id"
            )
        try:
            if suite_id == FIDELITY_SUITE_ID:
                result = run_fidelity_suite(
                    model_name=model,
                    seed=seed,
                    limit=limit,
                    condition_names=_parse_csv(conditions),
                )
            elif suite_id == PSYCHO_SUITE_ID:
                if conditions:
                    raise typer.BadParameter(
                        f"--conditions does not apply to {suite_id} (single condition set)"
                    )
                result = run_psycho_suite(model_name=model, seed=seed, limit=limit)
            else:
                if conditions:
                    raise typer.BadParameter(
                        f"--conditions does not apply to {suite_id} (single condition set)"
                    )
                result = run_phase_suite(model_name=model, seed=seed, limit=limit)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif suite_id == sound_id_suite.SUITE_ID:
        model_name = model or "heuristic-v0"
        custom_specs: list[MixtureSpec] = []
        if mix:
            custom_specs.extend(parse_inline_mix(list(mix)))
        if recipes is not None:
            custom_specs.extend(load_recipes(recipes))
        try:
            prompt_spec = load_prompts(prompts_file) if prompts_file is not None else load_prompts(None)
        except (PromptFormatError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        try:
            if json_out:
                result = sound_id_suite.run_suite(
                    model_name=model_name,
                    seed=seed,
                    pack_ids=_parse_csv(pack),
                    selected_conditions=_parse_csv(conditions),
                    profile_name=profile,
                    custom_mixtures=custom_specs or None,
                    limit=limit,
                    prompt_spec=prompt_spec,
                    prompt_ensemble=prompt_ensemble,
                )
            else:
                progress = _make_run_progress()
                with progress:
                    task_id = progress.add_task("preparing ab/sound-id", total=None)
                    result = sound_id_suite.run_suite(
                        model_name=model_name,
                        seed=seed,
                        pack_ids=_parse_csv(pack),
                        selected_conditions=_parse_csv(conditions),
                        profile_name=profile,
                        custom_mixtures=custom_specs or None,
                        limit=limit,
                        prompt_spec=prompt_spec,
                        prompt_ensemble=prompt_ensemble,
                        progress_callback=_sound_id_progress_callback(progress, task_id),
                    )
        except (ValueError, KeyError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        raise typer.BadParameter(f"unknown suite: {suite_id}")

    if output is None:
        output = Path("results") / f"run-{result['run_hash'][:8]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dump_json(result, pretty=True), encoding="utf-8")

    if json_out:
        console.print(_dump_json(result, pretty=pretty_json))
    else:
        render_run_summary(result, console=console)
        console.print(f"wrote: {output}")


@app.command("warmup")
def warmup(
    model: str = typer.Option("whisper-tiny", "--model", help="Whisper model to pre-download")
) -> None:
    normalized_model = model.replace("whisper-", "")
    warmup_model(normalized_model)
    console.print(f"model ready: whisper-{normalized_model}")


@app.command("compare")
def compare(
    run_a: Path,
    run_b: Path,
    pretty_json: bool = typer.Option(False, "--pretty-json", help="Pretty-print JSON output"),
    json_out: bool = typer.Option(False, "--json", help="Print JSON instead of a table"),
    allow_mismatched_prompt: bool = typer.Option(
        False,
        "--allow-mismatched-prompt",
        help="Permit comparing ab/sound-id runs with different prompt_version / parser_version / prompt_ensemble",
    ),
) -> None:
    if not run_a.exists():
        raise typer.BadParameter(f"file not found: {run_a}")
    if not run_b.exists():
        raise typer.BadParameter(f"file not found: {run_b}")
    suite, adapter = suite_from_run_json(run_a)
    with track_command("compare", suite=suite, adapter=adapter):
        _compare_body(
            run_a=run_a,
            run_b=run_b,
            pretty_json=pretty_json,
            json_out=json_out,
            allow_mismatched_prompt=allow_mismatched_prompt,
        )


def _compare_body(
    *,
    run_a: Path,
    run_b: Path,
    pretty_json: bool,
    json_out: bool,
    allow_mismatched_prompt: bool,
) -> None:
    left = json.loads(run_a.read_text(encoding="utf-8"))
    right = json.loads(run_b.read_text(encoding="utf-8"))
    try:
        summary = render_run_pair(
            left,
            right,
            console=console,
            allow_mismatched_prompt=allow_mismatched_prompt,
        )
    except CompareMismatchError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_out:
        console.print(_dump_json(summary, pretty=pretty_json))


@app.command("inspect")
def inspect(
    run_file: Path,
    mixture: int | None = typer.Option(
        None,
        "--mixture",
        help="Mixture index (1-based) — ab/sound-id only",
    ),
    clip: int | None = typer.Option(
        None,
        "--clip",
        help="Clip index (1-based) — ab/asr-robust and ab/asr-hallucination",
    ),
) -> None:
    if not run_file.exists():
        raise typer.BadParameter(f"file not found: {run_file}")
    data = json.loads(run_file.read_text(encoding="utf-8"))
    suite = data.get("suite")
    if suite in (ASR_SUITE_ID, ASR_HALLUCINATION_SUITE_ID):
        if mixture is not None:
            raise typer.BadParameter("--mixture only applies to ab/sound-id; use --clip for ASR runs")
        if clip is None:
            raise typer.BadParameter("--clip is required for ASR runs (1-based index)")
        _inspect_asr(data, clip)
        return
    if suite != sound_id_suite.SUITE_ID:
        raise typer.BadParameter(f"inspect does not support suite {suite!r}")
    if clip is not None:
        raise typer.BadParameter("--clip only applies to ASR runs; use --mixture for ab/sound-id")
    if mixture is None:
        raise typer.BadParameter("--mixture is required for ab/sound-id (1-based index)")
    per_mixture = data.get("per_mixture", [])
    if not per_mixture:
        raise typer.BadParameter("run JSON has no per_mixture records")
    if mixture < 1 or mixture > len(per_mixture):
        raise typer.BadParameter(f"mixture index out of range (1..{len(per_mixture)})")
    record = per_mixture[mixture - 1]
    console.print(
        f"mixture {mixture} (pack={record['pack']}, condition={record['condition']}, name={record['mixture_name']})"
    )
    components = record.get("components_present", [])
    console.print(f"  ground truth: {', '.join(components)}")
    console.print("  source clips:")
    for entry in record.get("sources", []):
        console.print(f"    {entry['label']:18} {entry['source']}")
    console.print()
    console.print(f"  model: {data.get('model')}")
    prompt_version = data.get("prompt_version")
    parser_version = data.get("parser_version")
    ensemble = data.get("prompt_ensemble")
    prompt_source = data.get("prompt_source")
    if prompt_version:
        ensemble_text = f"ensemble={ensemble}" if ensemble else "ensemble=off (single prompt)"
        source_text = f", source={prompt_source}" if prompt_source else ""
        console.print(
            f"  prompts: version={prompt_version}, parser={parser_version}, "
            f"{ensemble_text}{source_text}"
        )
    console.print("  yes responses:")
    yes_lines: list[str] = []
    no_lines: list[str] = []
    components_understood = 0
    components_total = len(components)
    yes_total = 0
    yes_correct = 0
    paraphrase_breakdowns: list[tuple[str, list[dict]]] = []
    for probe in record.get("probes", []):
        ans = probe.get("answered_yes")
        expected = probe.get("expected")
        paraphrases = probe.get("paraphrase_answers") or []
        if len(paraphrases) > 1:
            paraphrase_breakdowns.append((probe["label"], paraphrases))
        if ans:
            yes_total += 1
            if expected:
                yes_correct += 1
                components_understood += 1
                yes_lines.append(f"    {probe['label']:18} ✓")
            else:
                yes_lines.append(f"    {probe['label']:18} ✗  FALSE POSITIVE (distractor)")
        else:
            if expected:
                no_lines.append(f"    {probe['label']:18} ✗  FALSE NEGATIVE")
            else:
                no_lines.append(f"    {probe['label']:18} ✗  (distractor, correct)")
    for line in yes_lines:
        console.print(line)
    if no_lines:
        console.print("  no responses:")
        for line in no_lines:
            console.print(line)
    if paraphrase_breakdowns:
        console.print()
        console.print("  per-paraphrase breakdown:")
        for label, paraphrases in paraphrase_breakdowns:
            yes_count = sum(1 for p in paraphrases if p.get("answered_yes"))
            console.print(f"    {label:18} ({yes_count}/{len(paraphrases)} yes)")
            for entry in paraphrases:
                marker = "yes" if entry.get("answered_yes") else "no "
                color = "green" if entry.get("answered_yes") else "red"
                console.print(f"      \\[[{color}]{marker}[/{color}]] {entry.get('prompt')}")
    recall = components_understood / components_total if components_total else 0.0
    precision = yes_correct / yes_total if yes_total else 0.0
    console.print()
    console.print(f"  recall    : {components_understood}/{components_total} = {recall:.2f}")
    console.print(f"  precision : {yes_correct}/{yes_total} = {precision:.2f}")
    console.print(f"  components understood: {components_understood} of {components_total}")


def _inspect_asr(data: dict[str, Any], clip: int) -> None:
    from audiobench.metrics import compute_wer

    per_clip = data.get("per_clip_hypotheses", [])
    if not per_clip:
        raise typer.BadParameter("run JSON has no per_clip_hypotheses records")
    if clip < 1 or clip > len(per_clip):
        raise typer.BadParameter(f"clip index out of range (1..{len(per_clip)})")

    record = per_clip[clip - 1]
    suite = data.get("suite")
    clip_label = record.get("clip_id", record.get("file", clip))
    conditions: list[str] = list(data.get("conditions", []))
    if not conditions:
        conditions = list(record.get("hypotheses", {}).keys())
    reference = record.get("reference", "")

    header = f"clip {clip} (suite={suite}, id={clip_label}"
    if "file" in record:
        header += f", file={record['file']}"
    header += ")"
    console.print(header)
    console.print(f"  model: {data.get('model')} · seed={data.get('seed')}")
    metadata = record.get("metadata") or {}
    if metadata:
        meta_bits = []
        for key in ("domain", "duration_s", "noise_type", "snr_db", "language", "accent"):
            value = metadata.get(key)
            if value is None:
                continue
            meta_bits.append(f"{key}={value}")
        if meta_bits:
            console.print("  metadata: " + ", ".join(meta_bits))
    console.print()
    console.print(f"  reference: {reference!r}")
    console.print()

    hypotheses = record.get("hypotheses", {}) or {}
    condition_details = record.get("condition_details", {}) or {}

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("condition")
    table.add_column("hypothesis")
    table.add_column("clip WER", justify="right")
    table.add_column("latency ms", justify="right")
    table.add_column("flags")

    for condition in conditions:
        if condition not in hypotheses:
            continue
        hypothesis = hypotheses.get(condition, "")
        try:
            clip_wer = compute_wer([reference], [hypothesis])
        except Exception:
            clip_wer = float("nan")
        details = condition_details.get(condition) or {}
        latency = details.get("latency_ms")
        flags: list[str] = []
        if details.get("error"):
            flags.append("[red]error[/red]")
        if not hypothesis.strip():
            flags.append("[yellow]empty[/yellow]")
        if reference == "" and hypothesis.strip():
            flags.append("[red]hallucination[/red]")
        wer_text = "—" if clip_wer != clip_wer else f"{clip_wer:.1f}"
        latency_text = "—" if latency is None else f"{float(latency):.0f}"
        table.add_row(condition, hypothesis, wer_text, latency_text, " ".join(flags))
    console.print(table)

    if suite == ASR_SUITE_ID:
        per_condition_wer = data.get("per_condition_wer") or {}
        console.print()
        console.print(
            "  suite-level WER: "
            + ", ".join(
                f"{c}={per_condition_wer.get(c, 0.0):.2f}" for c in conditions if c in per_condition_wer
            )
        )
        weighted = data.get("weighted_mean_wer")
        if weighted is not None:
            console.print(f"  weighted mean WER: {weighted:.2f}")
    elif suite == ASR_HALLUCINATION_SUITE_ID:
        headline = data.get("headline") or {}
        if headline:
            console.print()
            console.print(
                "  suite-level: "
                f"hallucination={headline.get('non_speech_hallucination_rate', 0.0):.2f}, "
                f"empty={headline.get('non_speech_empty_rate', 0.0):.2f}, "
                f"mean inserted tokens={headline.get('non_speech_mean_inserted_tokens', 0.0):.2f}"
            )


@app.command("gate")
def gate(
    run_file: Path,
    thresholds: Path | None = typer.Option(
        None,
        "--thresholds",
        help="YAML or JSON file with per-suite thresholds (see docs).",
    ),
    max_weighted_mean_wer: float | None = typer.Option(
        None,
        "--max-wer",
        help="ab/asr-robust: cap on weighted mean WER (lower is better).",
    ),
    max_wer_condition: list[str] = typer.Option(
        [],
        "--max-wer-condition",
        help="ab/asr-robust: per-condition WER cap as CONDITION=VALUE (repeatable).",
    ),
    max_weighted_hallucination_rate: float | None = typer.Option(
        None,
        "--max-hallucination-rate",
        help="ab/asr-hallucination: cap on weighted hallucination rate.",
    ),
    max_non_speech_hallucination_rate: float | None = typer.Option(
        None,
        "--max-non-speech-hallucination-rate",
        help="ab/asr-hallucination: cap on overall non-speech hallucination rate.",
    ),
    min_weighted_recall: float | None = typer.Option(
        None,
        "--min-recall",
        help="ab/sound-id: floor on weighted recall (higher is better).",
    ),
    max_weighted_fpr: float | None = typer.Option(
        None,
        "--max-fpr",
        help="ab/sound-id: cap on weighted false-positive rate.",
    ),
    min_components_understood: int | None = typer.Option(
        None,
        "--min-components-understood",
        help="ab/sound-id: floor on absolute components-understood count.",
    ),
    min_weighted_si_sdr_db: float | None = typer.Option(
        None,
        "--min-si-sdr",
        help="ab/fidelity-roundtrip: floor on weighted SI-SDR (dB).",
    ),
    max_true_peak_dbtp: float | None = typer.Option(
        None,
        "--max-true-peak",
        help="ab/fidelity-roundtrip: cap on max true peak (dBTP).",
    ),
    min_masking_respect_score: float | None = typer.Option(
        None,
        "--min-masking-respect",
        help="ab/psychoacoustic-masking: floor on masking respect score (0-1).",
    ),
    min_phase_coherence_score: float | None = typer.Option(
        None,
        "--min-phase-coherence",
        help="ab/phase-coherence: floor on phase coherence score (0-1).",
    ),
    min_mean_polarity_score: float | None = typer.Option(
        None,
        "--min-polarity",
        help="ab/phase-coherence: floor on mean polarity score (0-1).",
    ),
    min_event_f1: float | None = typer.Option(
        None,
        "--min-event-f1",
        help="ab/sed-urban: floor on event-F1 at IoU=0.5 (0-1).",
    ),
    min_segment_f1: float | None = typer.Option(
        None,
        "--min-segment-f1",
        help="ab/sed-urban: floor on 1-second segment F1 (0-1).",
    ),
    max_der: float | None = typer.Option(
        None,
        "--max-der",
        help="ab/diarization-cw: ceiling on Diarization Error Rate (0-1).",
    ),
    max_speaker_count_error: float | None = typer.Option(
        None,
        "--max-speaker-count-error",
        help="ab/diarization-cw: ceiling on mean abs speaker-count error.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of a table."),
    pretty_json: bool = typer.Option(False, "--pretty-json", help="Pretty-print JSON output."),
    junit: Path | None = typer.Option(
        None,
        "--junit",
        help="Also write a JUnit XML report (one testcase per threshold check) to this path.",
    ),
) -> None:
    """Evaluate thresholds against a run JSON; exit non-zero on failure (CI-friendly)."""
    if not run_file.exists():
        raise typer.BadParameter(f"file not found: {run_file}")
    suite, adapter = suite_from_run_json(run_file)
    with track_command("gate", suite=suite, adapter=adapter):
        _gate_body(
            run_file=run_file,
            thresholds=thresholds,
            max_weighted_mean_wer=max_weighted_mean_wer,
            max_wer_condition=max_wer_condition,
            max_weighted_hallucination_rate=max_weighted_hallucination_rate,
            max_non_speech_hallucination_rate=max_non_speech_hallucination_rate,
            min_weighted_recall=min_weighted_recall,
            max_weighted_fpr=max_weighted_fpr,
            min_components_understood=min_components_understood,
            min_weighted_si_sdr_db=min_weighted_si_sdr_db,
            max_true_peak_dbtp=max_true_peak_dbtp,
            min_masking_respect_score=min_masking_respect_score,
            min_phase_coherence_score=min_phase_coherence_score,
            min_mean_polarity_score=min_mean_polarity_score,
            min_event_f1=min_event_f1,
            min_segment_f1=min_segment_f1,
            max_der=max_der,
            max_speaker_count_error=max_speaker_count_error,
            json_out=json_out,
            pretty_json=pretty_json,
            junit=junit,
        )


def _gate_body(
    *,
    run_file: Path,
    thresholds: Path | None,
    max_weighted_mean_wer: float | None,
    max_wer_condition: list[str],
    max_weighted_hallucination_rate: float | None,
    max_non_speech_hallucination_rate: float | None,
    min_weighted_recall: float | None,
    max_weighted_fpr: float | None,
    min_components_understood: int | None,
    min_weighted_si_sdr_db: float | None,
    max_true_peak_dbtp: float | None,
    min_masking_respect_score: float | None,
    min_phase_coherence_score: float | None,
    min_mean_polarity_score: float | None,
    min_event_f1: float | None,
    min_segment_f1: float | None,
    max_der: float | None,
    max_speaker_count_error: float | None,
    json_out: bool,
    pretty_json: bool,
    junit: Path | None,
) -> None:
    try:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid run JSON {run_file}: {exc}") from exc

    file_spec: dict[str, Any] | None = None
    if thresholds is not None:
        try:
            file_spec = load_thresholds(thresholds)
        except GateConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc

    inline_max_wer: dict[str, float] = {}
    for entry in max_wer_condition:
        if "=" not in entry:
            raise typer.BadParameter(
                f"--max-wer-condition expects CONDITION=VALUE, got {entry!r}"
            )
        name, raw_value = entry.split("=", 1)
        name = name.strip()
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"--max-wer-condition value must be numeric, got {raw_value!r}"
            ) from exc
        inline_max_wer[name] = value

    inline: dict[str, Any] = {
        "max_weighted_mean_wer": max_weighted_mean_wer,
        "max_weighted_hallucination_rate": max_weighted_hallucination_rate,
        "max_non_speech_hallucination_rate": max_non_speech_hallucination_rate,
        "min_weighted_recall": min_weighted_recall,
        "max_weighted_fpr": max_weighted_fpr,
        "min_components_understood": min_components_understood,
        "min_weighted_si_sdr_db": min_weighted_si_sdr_db,
        "max_true_peak_dbtp": max_true_peak_dbtp,
        "min_masking_respect_score": min_masking_respect_score,
        "min_phase_coherence_score": min_phase_coherence_score,
        "min_mean_polarity_score": min_mean_polarity_score,
        "min_event_f1": min_event_f1,
        "min_segment_f1": min_segment_f1,
        "max_der": max_der,
        "max_speaker_count_error": max_speaker_count_error,
    }
    if inline_max_wer:
        inline["max_wer"] = inline_max_wer

    suite = payload.get("suite", "")
    spec = merge_inline_thresholds(file_spec, suite=suite, inline=inline)

    try:
        report = evaluate_gate(payload, spec)
    except GateConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_out:
        typer.echo(_dump_json(report.to_dict(), pretty=pretty_json))
    else:
        for line in format_gate_console(report):
            console.print(line)

    if junit is not None:
        junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_text(render_junit_xml([gate_report_to_junit(report)]), encoding="utf-8")

    if not report.passed:
        raise typer.Exit(code=1)


def _run_cell(cell: MatrixCell, seed: int) -> dict[str, Any]:
    """Dispatch a single :class:`MatrixCell` to its suite runner."""
    if cell.suite in asr_suite_ids():
        if cell.profile or cell.pack:
            raise ValueError(
                f"{cell.display_name()}: --profile/--pack only apply to ab/sound-id"
            )
        if cell.suite == ASR_SUITE_ID:
            return run_asr_suite(
                model_name=cell.model,
                seed=seed,
                limit=cell.limit,
                condition_names=cell.conditions,
            )
        return run_asr_hallucination_suite(
            model_name=cell.model,
            seed=seed,
            limit=cell.limit,
            condition_names=cell.conditions,
        )
    if cell.suite == sound_id_suite.SUITE_ID:
        prompt_spec = load_prompts(None)
        return sound_id_suite.run_suite(
            model_name=cell.model,
            seed=seed,
            pack_ids=cell.pack,
            selected_conditions=cell.conditions,
            profile_name=cell.profile,
            limit=cell.limit,
            prompt_spec=prompt_spec,
            prompt_ensemble=None,
        )
    if cell.suite in signal_suite_ids():
        if cell.profile or cell.pack:
            raise ValueError(
                f"{cell.display_name()}: --profile/--pack only apply to ab/sound-id"
            )
        if cell.suite == FIDELITY_SUITE_ID:
            return run_fidelity_suite(
                model_name=cell.model,
                seed=seed,
                limit=cell.limit,
                condition_names=cell.conditions,
            )
        if cell.suite == PSYCHO_SUITE_ID:
            return run_psycho_suite(model_name=cell.model, seed=seed, limit=cell.limit)
        return run_phase_suite(model_name=cell.model, seed=seed, limit=cell.limit)
    if cell.suite in temporal_suite_ids():
        if cell.profile or cell.pack:
            raise ValueError(
                f"{cell.display_name()}: --profile/--pack only apply to ab/sound-id"
            )
        if cell.suite == SED_SUITE_ID:
            return run_sed_suite(model_name=cell.model, seed=seed, limit=cell.limit)
        return run_diar_suite(model_name=cell.model, seed=seed, limit=cell.limit)
    raise ValueError(f"{cell.display_name()}: unknown suite {cell.suite!r}")


def _gate_cell(cell_result: CellResult, gate_spec: dict[str, Any]) -> dict[str, Any] | None:
    """Apply ``gate_spec`` to a single cell's run JSON if present."""
    if cell_result.status != "ok" or cell_result.output_path is None:
        return None
    try:
        payload = json.loads(Path(cell_result.output_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": f"failed to read run JSON: {exc}"}
    try:
        report = evaluate_gate(payload, gate_spec)
    except GateConfigError:
        # No thresholds apply to this suite; skip silently.
        return None
    return report.to_dict()


@app.command("run-matrix")
def run_matrix(
    matrix: Path | None = typer.Option(
        None,
        "--matrix",
        help="YAML/JSON matrix file (overrides --suite/--model).",
    ),
    suites: list[str] = typer.Option(
        [],
        "--suite",
        help="Suite id (repeatable). Combined with --model as a cartesian product.",
    ),
    models: list[str] = typer.Option(
        [],
        "--model",
        help="Model adapter id (repeatable).",
    ),
    output_dir: Path = typer.Option(
        Path("results/matrix"),
        "--output-dir",
        help="Directory for per-cell run JSONs and the aggregated summary.",
    ),
    seed: int = typer.Option(1337, "--seed", help="Default seed for cells without one."),
    limit: int | None = typer.Option(None, "--limit", help="Default clip/mixture limit per cell."),
    conditions: str | None = typer.Option(
        None, "--conditions", help="Default comma-separated conditions per cell."
    ),
    pack: str | None = typer.Option(
        None, "--pack", help="Default comma-separated packs (ab/sound-id only)."
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Default profile (ab/sound-id only)."
    ),
    gate_file: Path | None = typer.Option(
        None,
        "--gate",
        help="Threshold YAML/JSON applied per cell after each run.",
    ),
    junit: Path | None = typer.Option(
        None,
        "--junit",
        help="Write aggregated JUnit XML (one testcase per cell, plus gate checks if --gate).",
    ),
    summary_name: str = typer.Option(
        "summary.json", "--summary-name", help="Filename for the aggregated summary inside --output-dir."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the aggregated summary as JSON."),
    pretty_json: bool = typer.Option(False, "--pretty-json", help="Pretty-print JSON output."),
) -> None:
    """Run several (suite, model) cells in one shot; aggregate results for CI."""
    if matrix is not None:
        try:
            plan = load_matrix_file(matrix)
        except MatrixConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if output_dir != Path("results/matrix"):
            plan.output_dir = output_dir
        if seed != 1337:
            plan.seed = seed
    else:
        try:
            cells = build_cells_from_cli(
                suites,
                models,
                limit=limit,
                conditions=_parse_csv(conditions),
                pack=_parse_csv(pack),
                profile=profile,
            )
        except MatrixConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        plan = MatrixPlan(
            cells=cells,
            output_dir=output_dir,
            seed=seed,
            gate_spec=None,
        )

    gate_spec = plan.gate_spec
    if gate_file is not None:
        try:
            file_spec = load_thresholds(gate_file)
        except GateConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        gate_spec = {**(gate_spec or {}), **file_spec}

    results = run_matrix_cells(plan, runner=_run_cell)

    gate_status: dict[str, dict[str, Any]] = {}
    if gate_spec:
        for result in results:
            report = _gate_cell(result, gate_spec)
            if report is not None:
                gate_status[result.cell.display_name()] = report

    summary = build_summary(plan, results, gate_status=gate_status)
    summary_path = plan.output_dir / summary_name
    summary_path.write_text(_dump_json(summary, pretty=True), encoding="utf-8")

    if json_out:
        typer.echo(_dump_json(summary, pretty=pretty_json))
    else:
        _render_matrix_console(summary)
        console.print(f"wrote summary: {summary_path}")

    if junit is not None:
        junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_text(_matrix_junit(summary), encoding="utf-8")

    if summary["error_count"] > 0 or summary["gate_failed_count"] > 0:
        raise typer.Exit(code=1)


def _render_matrix_console(summary: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("#", justify="right")
    table.add_column("cell")
    table.add_column("status")
    table.add_column("headline")
    table.add_column("gate")
    table.add_column("duration s", justify="right")
    for entry in summary["cells"]:
        cell = entry["cell"]
        name = cell.get("name") or f"{cell['suite']}::{cell['model']}"
        status = entry["status"]
        status_text = "[green]ok[/green]" if status == "ok" else f"[red]{status}[/red]"
        headline_text = _summarize_headline(cell.get("suite", ""), entry.get("headline") or {}, entry.get("error"))
        gate = entry.get("gate")
        if gate is None:
            gate_text = "—"
        elif gate.get("passed"):
            gate_text = "[green]pass[/green]"
        else:
            failed = gate.get("failure_count") or len(gate.get("checks") or [])
            gate_text = f"[red]fail[/red] ({failed})"
        table.add_row(
            str(entry["index"]),
            name,
            status_text,
            headline_text,
            gate_text,
            f"{entry['duration_s']:.2f}",
        )
    console.print(table)
    console.print(
        f"cells={summary['cell_count']}  ok={summary['ok_count']}  "
        f"error={summary['error_count']}  gate_failed={summary['gate_failed_count']}"
    )


def _summarize_headline(suite: str, headline: dict[str, Any], error: str | None) -> str:
    if error:
        return f"[red]{error}[/red]"
    if suite == "ab/asr-robust":
        value = headline.get("weighted_mean_wer")
        return f"WER={value:.2f}" if isinstance(value, (int, float)) else "—"
    if suite == "ab/asr-hallucination":
        value = headline.get("weighted_hallucination_rate")
        finding = headline.get("top_finding_status")
        body = f"hallu={value:.2f}" if isinstance(value, (int, float)) else "—"
        if finding:
            body += f" · finding={finding}"
        return body
    if suite == "ab/sound-id":
        recall = headline.get("weighted_recall")
        fpr = headline.get("weighted_fpr")
        understood = headline.get("components_understood")
        present = headline.get("components_present")
        parts: list[str] = []
        if isinstance(recall, (int, float)):
            parts.append(f"recall={recall:.2f}")
        if isinstance(fpr, (int, float)):
            parts.append(f"fpr={fpr:.2f}")
        if understood is not None and present is not None:
            parts.append(f"{understood}/{present}")
        return " · ".join(parts) if parts else "—"
    if suite == "ab/fidelity-roundtrip":
        sdr = headline.get("weighted_si_sdr_db")
        tp = headline.get("max_true_peak_dbtp")
        parts = []
        if isinstance(sdr, (int, float)):
            parts.append(f"SI-SDR={sdr:.2f}dB")
        if isinstance(tp, (int, float)):
            parts.append(f"TP={tp:+.2f}dBTP")
        return " · ".join(parts) if parts else "—"
    if suite == "ab/psychoacoustic-masking":
        score = headline.get("masking_respect_score")
        count = headline.get("respected_count")
        total = headline.get("stimulus_count")
        if isinstance(score, (int, float)):
            return f"respect={score:.2f} ({count}/{total})"
        return "—"
    if suite == "ab/phase-coherence":
        score = headline.get("phase_coherence_score")
        polarity = headline.get("mean_polarity_score")
        parts = []
        if isinstance(score, (int, float)):
            parts.append(f"coherence={score:.2f}")
        if isinstance(polarity, (int, float)):
            parts.append(f"polarity={polarity:.2f}")
        return " · ".join(parts) if parts else "—"
    return "—"


def _matrix_junit(summary: dict[str, Any]) -> str:
    classname = "audiobench.matrix"
    cell_suite = JUnitSuite(name=classname)
    gate_suite = JUnitSuite(name=f"{classname}.gate")
    for entry in summary["cells"]:
        cell = entry["cell"]
        name = cell.get("name") or f"{cell['suite']}::{cell['model']}"
        if entry["status"] == "ok":
            cell_suite.cases.append(JUnitCase(classname=classname, name=name))
        else:
            cell_suite.cases.append(
                JUnitCase(
                    classname=classname,
                    name=name,
                    failure_message=entry.get("error") or "cell failed",
                )
            )
        gate = entry.get("gate")
        if gate is None:
            continue
        for check in gate.get("checks", []):
            check_name = f"{name} · {check['name']}"
            if check["passed"]:
                gate_suite.cases.append(JUnitCase(classname=f"{classname}.gate", name=check_name))
            else:
                actual = "missing" if check.get("actual") is None else f"{check['actual']:g}"
                msg = (
                    f"actual={actual} {check['comparator']} threshold={check['threshold']:g} failed"
                )
                if check.get("detail"):
                    msg += f" ({check['detail']})"
                gate_suite.cases.append(
                    JUnitCase(
                        classname=f"{classname}.gate",
                        name=check_name,
                        failure_message=msg,
                    )
                )
    suites = [cell_suite]
    if gate_suite.tests:
        suites.append(gate_suite)
    return render_junit_xml(suites)


@prompts_app.command("show")
def prompts_show(
    file: Path | None = typer.Option(
        None, "--file", help="Show a user prompts file instead of the bundled one"
    ),
) -> None:
    if file is None:
        text = bundled_prompts_text()
        console.print("[dim]# bundled: src/audiobench/data/sound_id/prompts.yaml[/dim]")
    else:
        if not file.exists():
            raise typer.BadParameter(f"file not found: {file}")
        text = file.read_text(encoding="utf-8")
        console.print(f"[dim]# {file}[/dim]")
    try:
        spec = load_prompts(file)
    except (PromptFormatError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"[bold]version[/bold]={spec.version}  [bold]parser[/bold]={spec.parser_version}  "
        f"[bold]paraphrases[/bold]={len(spec.paraphrases)}"
    )
    console.print(text)


@prompts_app.command("export")
def prompts_export(
    target: Path,
    force: bool = typer.Option(False, "--force", help="Overwrite if target exists"),
) -> None:
    try:
        out = export_default_prompts(target, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"wrote {out}")
    console.print(
        "edit it, then run with `audiobench run ab/sound-id --prompts "
        f"{out}` (use the same path/version when comparing)."
    )


def _resolve_mix_spec_for_preview(
    *,
    labels: str | None,
    recipes: Path | None,
    name: str | None,
    pack: str,
) -> tuple[MixtureSpec, dict[str, float]]:
    if labels:
        slugs = [item.strip() for item in labels.split(",") if item.strip()]
        spec = parse_inline_mix(["+".join(slugs)])[0]
        return spec, {}
    if recipes is None:
        raise typer.BadParameter("provide either --labels or --recipes")
    specs = load_recipes(recipes)
    if name:
        chosen = [s for s in specs if s.name == name]
        if not chosen:
            raise typer.BadParameter(f"recipe {name!r} not found in {recipes}")
        spec = chosen[0]
    elif len(specs) == 1:
        spec = specs[0]
    else:
        raise typer.BadParameter(
            f"recipe file has {len(specs)} mixtures; pass --name to choose one"
        )
    return spec, {label: level for label, level in spec.label_levels}


@mix_app.command("preview")
def mix_preview(
    output: Path = typer.Option(..., "--output", help="Path for the rendered mixture WAV"),
    labels: str | None = typer.Option(None, "--labels", help="Comma-separated labels for an inline mix"),
    recipes: Path | None = typer.Option(None, "--recipes", help="YAML/JSON recipe file"),
    name: str | None = typer.Option(None, "--name", help="Recipe entry name to render"),
    pack: str = typer.Option("demo", "--pack", help="Pack to source clips from"),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed"),
) -> None:
    spec, label_levels = _resolve_mix_spec_for_preview(
        labels=labels, recipes=recipes, name=name, pack=pack
    )
    manifest = load_pack_manifest(pack)
    resolver = make_resolver(manifest)
    if isinstance(resolver, UserCacheResolver) and not resolver.is_available():
        raise typer.BadParameter(
            f"pack {pack} requires data at {resolver.directory} (expected: {manifest.expected_layout})"
        )
    sources: list[MixSource] = []
    for label in spec.labels:
        if label not in manifest.labels:
            raise typer.BadParameter(
                f"label {label!r} not in pack {pack!r}; pack labels: {', '.join(manifest.labels)}"
            )
        variant = (int(sha256_text(f"{seed}|{spec.name}|{label}")[:8], 16)) % max(1, resolver.variants_for(label))
        audio = resolver.load(label, variant)
        sources.append(MixSource(label=label, audio=audio, sample_rate=resolver.sample_rate))
    mixture, sr = mix_sources(sources, snr_db=spec.snr_db, label_levels=label_levels or None)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), np.asarray(mixture, dtype=np.float32), sr)
    console.print(f"wrote {output} ({len(mixture) / sr:.2f}s @ {sr} Hz, labels={list(spec.labels)})")


@app.command("push")
def push(
    run_file: Path,
    repo: str | None = typer.Option(
        None,
        "--repo",
        envvar=DEFAULT_DATASET_ENV,
        help=(
            "Hugging Face dataset repo id. Defaults to "
            "<your-username>/audiobench-leaderboard-submissions from your hf auth login "
            f"(or set ${DEFAULT_DATASET_ENV})."
        ),
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        envvar="HF_TOKEN",
        help="Optional Hugging Face token (falls back to the active `hf auth login` token).",
    ),
    space: str | None = typer.Option(
        None,
        "--space",
        envvar=DEFAULT_SPACE_ENV,
        help=f"Optional leaderboard Space id to print (or ${DEFAULT_SPACE_ENV})",
    ),
    notes: str | None = typer.Option(None, "--notes", help="Optional notes attached to the submission"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags for filtering on the leaderboard"),
    author: str | None = typer.Option(
        None,
        "--author",
        envvar="AUDIOBENCH_AUTHOR",
        help="Benchmark author (e.g. 'Phonon'). Recorded as authored_by; distinct from the HF uploader.",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Create the dataset repo as private if it does not already exist",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite an existing submission with the same run_hash",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build and print the submission payload without uploading"),
    pretty_json: bool = typer.Option(True, "--pretty-json/--compact-json", help="Pretty-print JSON output"),
) -> None:
    if not run_file.exists():
        raise typer.BadParameter(f"file not found: {run_file}")
    suite, adapter = suite_from_run_json(run_file)
    with track_command("push", suite=suite, adapter=adapter):
        _push_body(
            run_file=run_file,
            repo=repo,
            token=token,
            space=space,
            notes=notes,
            tags=tags,
            author=author,
            private=private,
            overwrite=overwrite,
            dry_run=dry_run,
            pretty_json=pretty_json,
        )


def _push_body(
    *,
    run_file: Path,
    repo: str | None,
    token: str | None,
    space: str | None,
    notes: str | None,
    tags: str | None,
    author: str | None,
    private: bool,
    overwrite: bool,
    dry_run: bool,
    pretty_json: bool,
) -> None:
    shared_repo = "THENIROCK/audiobench-leaderboard-submissions"
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    submission_tags = _parse_csv(tags)
    resolved_repo = repo
    repo_autodetected = False

    if dry_run:
        if resolved_repo is None:
            try:
                resolved_repo = default_dataset_repo_id(token=token)
                repo_autodetected = True
            except RuntimeError:
                resolved_repo = None
        try:
            record = build_submission_record(
                payload,
                authored_by=author,
                notes=notes,
                tags=submission_tags,
                source_file=str(run_file),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        preview = {
            "suite": record["suite"],
            "revision": record["revision"],
            "model": record["model"],
            "run_hash": record["run_hash"],
            "payload_sha256": record["payload_sha256"],
            "authored_by": record.get("authored_by"),
            "mode": "dry-run",
            "repo": resolved_repo,
            "repo_autodetected": repo_autodetected,
            "space": space,
            "path_preview": f"submissions/{record['suite'].replace('/', '__')}/{record['run_hash']}.json",
        }
        metrics = (record.get("leaderboard") or {}).get("metrics") or {}
        if metrics.get("top_finding_status") is not None:
            preview["top_finding_status"] = metrics.get("top_finding_status")
            preview["validated_findings"] = metrics.get("validated_findings")
        console.print(_dump_json(preview, pretty=pretty_json))
        console.print("dry run complete; no data was sent.")
        return

    if resolved_repo is None:
        try:
            resolved_repo = default_dataset_repo_id(token=token)
            repo_autodetected = True
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc

    try:
        record, result = push_submission(
            payload,
            repo_id=resolved_repo,
            token=token,
            private=private,
            allow_overwrite=overwrite,
            authored_by=author,
            notes=notes,
            tags=submission_tags,
            source_file=str(run_file),
            space_id=space,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        raise typer.BadParameter(f"push failed: {exc}") from exc

    summary = {
        "suite": record["suite"],
        "revision": record["revision"],
        "model": record["model"],
        "run_hash": record["run_hash"],
        "payload_sha256": record["payload_sha256"],
        "authored_by": record.get("authored_by"),
        "repo": result.repo_id,
        "path_in_repo": result.path_in_repo,
        "uploaded": result.uploaded,
        "duplicate": result.duplicate,
        "repo_autodetected": repo_autodetected,
        "dataset_url": result.dataset_url,
        "space_url": result.space_url,
    }
    metrics = (record.get("leaderboard") or {}).get("metrics") or {}
    if metrics.get("top_finding_status") is not None:
        summary["top_finding_status"] = metrics.get("top_finding_status")
        summary["validated_findings"] = metrics.get("validated_findings")
    console.print(_dump_json(summary, pretty=pretty_json))
    if result.duplicate and not overwrite:
        console.print(
            "submission already exists for this run_hash; pass --overwrite to replace it in the dataset."
        )
    else:
        console.print("uploaded submission to Hugging Face dataset leaderboard store.")
    if repo_autodetected and resolved_repo != shared_repo:
        console.print(f"Tip: push to the public leaderboard with --repo {shared_repo}")


@app.command()
def demo() -> None:
    """Interactive demo mode — YC pitch."""
    from audiobench.demo import run_demo

    run_demo()


if __name__ == "__main__":
    app()
