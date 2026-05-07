from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.table import Table

from audiobench.compare import CompareMismatchError, render_run_pair
from audiobench.hashing import sha256_text, stable_json
from audiobench.mixing import MixSource, mix_sources
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
from audiobench.suites import sound_id as sound_id_suite
from audiobench.suites.asr_robust import (
    SUITE_ID as ASR_SUITE_ID,
    SUITE_REVISION as ASR_SUITE_REVISION,
    load_manifest as load_asr_manifest,
    run_suite as run_asr_suite,
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
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(PHONON_LOGO)
        console.print("  [dim]run `audiobench --help` for commands[/]")


def _dump_json(data: dict[str, Any], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True)
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def _parse_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


@app.command("list")
def list_suites() -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("suite id")
    table.add_column("domain")
    table.add_column("tasks", justify="right")
    table.add_column("status")
    table.add_row("ab/asr-robust", "speech recognition", "5", "stable (MVP)")
    table.add_row("ab/sound-id", "sound event identification", "4", "stable (MVP)")
    table.add_row("ab/separation-musdb+", "source separation", "6", "in design")
    table.add_row("ab/tagging-audioset-v2", "audio tagging", "9", "in design")
    table.add_row("ab/diarization-cw", "speaker diarization", "5", "in design")
    table.add_row("ab/music-tag-mtg", "music understanding", "8", "in design")
    table.add_row("ab/sed-urban", "sound event detection", "7", "in design")
    table.add_row("ab/codec-perceptual", "neural codecs", "4", "in design")
    table.add_row("ab/tts-eval", "speech synthesis", "6", "in design")
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
        return
    raise typer.BadParameter(f"unknown suite: {suite_id}")


@app.command("run")
def run(
    suite_id: str,
    model: str = typer.Option("tiny", "--model", help="Model adapter name"),
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
    if suite_id == ASR_SUITE_ID:
        if mix or recipes or pack or profile or prompts_file or prompt_ensemble is not None:
            raise typer.BadParameter(
                "--mix/--recipes/--pack/--profile/--prompts/--prompt-ensemble only apply to ab/sound-id"
            )
        normalized_model = model.replace("whisper-", "")
        try:
            result = run_asr_suite(
                model_name=normalized_model,
                seed=seed,
                limit=limit,
                condition_names=_parse_csv(conditions),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif suite_id == sound_id_suite.SUITE_ID:
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
            result = sound_id_suite.run_suite(
                model_name=model,
                seed=seed,
                pack_ids=_parse_csv(pack),
                selected_conditions=_parse_csv(conditions),
                profile_name=profile,
                custom_mixtures=custom_specs or None,
                limit=limit,
                prompt_spec=prompt_spec,
                prompt_ensemble=prompt_ensemble,
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
    mixture: int = typer.Option(..., "--mixture", help="Mixture index (1-based)"),
) -> None:
    if not run_file.exists():
        raise typer.BadParameter(f"file not found: {run_file}")
    data = json.loads(run_file.read_text(encoding="utf-8"))
    if data.get("suite") != sound_id_suite.SUITE_ID:
        raise typer.BadParameter("inspect only supports ab/sound-id runs")
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
    pretty_json: bool = typer.Option(True, "--pretty-json/--compact-json", help="Pretty-print JSON output"),
) -> None:
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
    console.print(_dump_json(signed_payload, pretty=pretty_json))
    console.print("push is a local stub in MVP mode; no data was sent.")


if __name__ == "__main__":
    app()
