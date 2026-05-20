"""ab/sound-id suite runner.

Mixes labeled clips from one or more packs, asks the model one or more
versioned yes/no prompts per probe, and reports per-pack / per-condition
recall, precision, F1, and FPR plus a headline "components understood" count.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from audiobench.hashing import run_hash, sha256_text, stable_json
from audiobench.mixing import MixSource, mix_sources
from audiobench.models.audio_llm import AudioLLMAdapter
from audiobench.models.registry import make_model
from audiobench.packs import (
    ClipResolver,
    PackManifest,
    UserCacheResolver,
    filter_to_available,
    list_pack_ids,
    load_pack_manifest,
    make_resolver,
)
from audiobench.probes import Probe, build_probes, majority_vote, parse_yes_no
from audiobench.profiles import Profile, get_profile
from audiobench.prompts import PromptSpec, load_prompts
from audiobench.recipes import MixtureSpec
from audiobench.sound_id_metrics import ProbeOutcome, aggregate, per_class_breakdown


SUITE_ID = "ab/sound-id"
SUITE_REVISION = "0.1.0"

CONDITIONS: list[tuple[str, int]] = [
    ("solo", 1),
    ("pair", 2),
    ("triple", 3),
    ("quad", 4),
]

CUSTOM_CONDITION = "custom"
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class _PackRuntime:
    manifest: PackManifest
    resolver: ClipResolver


def _resolve_pack_runtimes(pack_ids: Iterable[str]) -> tuple[list[_PackRuntime], list[tuple[str, str]]]:
    runtimes: list[_PackRuntime] = []
    skipped: list[tuple[str, str]] = []
    for pid in pack_ids:
        try:
            manifest = load_pack_manifest(pid)
        except KeyError as exc:
            skipped.append((pid, str(exc)))
            continue
        resolver = make_resolver(manifest)
        if isinstance(resolver, UserCacheResolver) and not resolver.is_available():
            skipped.append(
                (
                    pid,
                    f"missing data at {resolver.directory} (expected: {manifest.expected_layout})",
                )
            )
            continue
        runtimes.append(_PackRuntime(manifest=manifest, resolver=resolver))
    return runtimes, skipped


def _resolve_pack_filter(
    *,
    requested: list[str] | None,
    profile: Profile | None,
) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    if profile and profile.pack_filter:
        return list(profile.pack_filter)
    available = list_pack_ids()
    statuses = filter_to_available(available)
    return [pid for pid, ok, _ in statuses if ok] or ["demo"]


def _condition_counts(manifest: PackManifest, profile: Profile | None) -> dict[str, int]:
    if profile and profile.use_demo_fast_counts and manifest.demo_fast_mixture_counts:
        counts = dict(manifest.demo_fast_mixture_counts)
    else:
        counts = dict(manifest.mixture_counts)
    if profile and profile.condition_filter:
        allowed = set(profile.condition_filter)
        counts = {k: v for k, v in counts.items() if k in allowed}
    return counts


def _draw_default_mixtures(
    manifest: PackManifest,
    *,
    profile: Profile | None,
    selected_conditions: list[str] | None,
    seed: int,
) -> list[tuple[str, MixtureSpec]]:
    counts = _condition_counts(manifest, profile)
    if selected_conditions:
        counts = {k: v for k, v in counts.items() if k in set(selected_conditions)}
    rng = random.Random(seed ^ sha256_int(manifest.id))

    out: list[tuple[str, MixtureSpec]] = []
    for condition_name, n_components in CONDITIONS:
        if condition_name not in counts or counts[condition_name] <= 0:
            continue
        if n_components > len(manifest.labels):
            continue
        for index in range(counts[condition_name]):
            chosen = rng.sample(list(manifest.labels), n_components)
            spec = MixtureSpec(
                name=f"{manifest.id}-{condition_name}-{index + 1:03d}",
                labels=tuple(chosen),
                pack=manifest.id,
            )
            out.append((condition_name, spec))
    return out


def sha256_int(text: str) -> int:
    return int(sha256_text(text)[:12], 16)


def _select_clip_variants(
    *,
    spec: MixtureSpec,
    resolver: ClipResolver,
    seed: int,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    pin_lookup = {key: value for key, value in spec.pin}
    for label in spec.labels:
        if label in pin_lookup:
            out.append((label, _hash_to_variant(pin_lookup[label], resolver.variants_for(label))))
            continue
        variants = max(1, resolver.variants_for(label))
        out.append((label, rng.randrange(variants)))
    return out


def _hash_to_variant(value: str, variants: int) -> int:
    if variants <= 1:
        return 0
    return int(sha256_text(value)[:8], 16) % variants


def _label_levels_dict(spec: MixtureSpec) -> dict[str, float]:
    return {label: level for label, level in spec.label_levels}


def run_suite(
    *,
    model_name: str,
    seed: int = 1337,
    pack_ids: list[str] | None = None,
    selected_conditions: list[str] | None = None,
    profile_name: str | None = None,
    custom_mixtures: list[MixtureSpec] | None = None,
    limit: int | None = None,
    model: AudioLLMAdapter | None = None,
    prompt_spec: PromptSpec | None = None,
    prompt_ensemble: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    profile = get_profile(profile_name) if profile_name else None
    if prompt_spec is None:
        prompt_spec = load_prompts(None)
    if prompt_ensemble is not None and prompt_ensemble < 1:
        raise ValueError(f"prompt_ensemble must be >= 1, got {prompt_ensemble}")
    requested_packs = _resolve_pack_filter(requested=pack_ids, profile=profile)
    runtimes, skipped = _resolve_pack_runtimes(requested_packs)

    if not runtimes:
        raise ValueError(
            "no packs available. " + (
                f"skipped: {skipped!r}" if skipped else "use `audiobench list-packs` to inspect."
            )
        )

    runtime_by_pack = {rt.manifest.id: rt for rt in runtimes}

    adapter: AudioLLMAdapter = model if model is not None else make_model(model_name)

    rng_seed = int(seed)
    pack_results: dict[str, dict] = {}
    per_mixture_records: list[dict] = []
    custom_outcomes: list[ProbeOutcome] = []
    custom_records_count = 0
    default_mixtures_by_pack: dict[str, list[tuple[str, MixtureSpec]]] = {}
    if not custom_mixtures:
        for runtime in runtimes:
            seed_for_pack = rng_seed ^ sha256_int(runtime.manifest.id)
            default_mixtures = _draw_default_mixtures(
                runtime.manifest,
                profile=profile,
                selected_conditions=selected_conditions,
                seed=seed_for_pack,
            )
            if limit is not None:
                default_mixtures = default_mixtures[: max(1, int(limit))]
            default_mixtures_by_pack[runtime.manifest.id] = default_mixtures

    total_mixtures = (
        len(custom_mixtures or [])
        if custom_mixtures
        else sum(len(items) for items in default_mixtures_by_pack.values())
    )
    _emit_progress(
        progress_callback,
        "start",
        suite=SUITE_ID,
        model=adapter.name,
        packs=[rt.manifest.id for rt in runtimes],
        total_mixtures=total_mixtures,
    )
    mixture_index = 0

    if custom_mixtures:
        for index, spec in enumerate(custom_mixtures):
            mixture_index += 1
            spec_pack = spec.pack or runtimes[0].manifest.id
            if spec_pack not in runtime_by_pack:
                spec_pack = runtimes[0].manifest.id
            runtime = runtime_by_pack[spec_pack]
            resolver = runtime.resolver
            present = list(spec.labels)
            mixture_seed = rng_seed ^ sha256_int(f"custom|{index}|{spec.name}")
            variants = _select_clip_variants(spec=spec, resolver=resolver, seed=mixture_seed)
            sources = []
            source_ids = []
            for label, variant in variants:
                audio = resolver.load(label, variant)
                sources.append(MixSource(label=label, audio=audio, sample_rate=resolver.sample_rate))
                source_ids.append({"label": label, "source": resolver.source_id(label, variant)})
            mixture_audio, mixture_sr = mix_sources(
                sources, snr_db=spec.snr_db, label_levels=_label_levels_dict(spec)
            )
            _emit_progress(
                progress_callback,
                "mixture_start",
                pack=spec_pack,
                condition=CUSTOM_CONDITION,
                mixture_name=spec.name,
                mixture_index=mixture_index,
                total_mixtures=total_mixtures,
            )
            probes = build_probes(
                spec=prompt_spec,
                present=present,
                pack_labels=list(runtime.manifest.labels),
                distractor_count=runtime.manifest.distractor_count,
                seed=mixture_seed ^ 0xA5A5,
                ensemble=prompt_ensemble,
            )
            outcomes, probe_records = _run_probes(
                adapter,
                mixture_audio,
                mixture_sr,
                probes,
                progress_callback=progress_callback,
                pack=spec_pack,
                condition=CUSTOM_CONDITION,
                mixture_name=spec.name,
            )
            custom_outcomes.extend(outcomes)
            custom_records_count += 1
            per_mixture_records.append(
                {
                    "pack": spec_pack,
                    "condition": CUSTOM_CONDITION,
                    "mixture_name": spec.name,
                    "spec": spec.to_dict(),
                    "components_present": present,
                    "sources": source_ids,
                    "probes": probe_records,
                }
            )
            _emit_progress(
                progress_callback,
                "mixture_done",
                pack=spec_pack,
                condition=CUSTOM_CONDITION,
                mixture_name=spec.name,
                mixture_index=mixture_index,
                total_mixtures=total_mixtures,
            )

    skip_default_mixtures = bool(custom_mixtures)

    for runtime in runtimes:
        manifest = runtime.manifest
        if skip_default_mixtures:
            pack_results.setdefault(
                manifest.id,
                {
                    "manifest": _serializable_manifest(manifest),
                    "per_condition": {},
                    "per_class": {},
                    "totals": {
                        "components_present": 0.0,
                        "components_understood": 0.0,
                        "tp": 0.0,
                        "fp": 0.0,
                        "fn": 0.0,
                        "tn": 0.0,
                        "recall": 0.0,
                        "precision": 0.0,
                        "f1": 0.0,
                        "fpr": 0.0,
                    },
                },
            )
            continue

        seed_for_pack = rng_seed ^ sha256_int(manifest.id)
        default_mixtures = default_mixtures_by_pack.get(manifest.id, [])

        per_condition_outcomes: dict[str, list[ProbeOutcome]] = {}
        all_outcomes: list[ProbeOutcome] = []
        for condition_name, spec in default_mixtures:
            mixture_index += 1
            mixture_seed = seed_for_pack ^ sha256_int(spec.name)
            variants = _select_clip_variants(spec=spec, resolver=runtime.resolver, seed=mixture_seed)
            sources = []
            source_ids = []
            for label, variant in variants:
                audio = runtime.resolver.load(label, variant)
                sources.append(MixSource(label=label, audio=audio, sample_rate=runtime.resolver.sample_rate))
                source_ids.append({"label": label, "source": runtime.resolver.source_id(label, variant)})
            mixture_audio, mixture_sr = mix_sources(
                sources, snr_db=spec.snr_db, label_levels=_label_levels_dict(spec)
            )
            _emit_progress(
                progress_callback,
                "mixture_start",
                pack=manifest.id,
                condition=condition_name,
                mixture_name=spec.name,
                mixture_index=mixture_index,
                total_mixtures=total_mixtures,
            )
            probes = build_probes(
                spec=prompt_spec,
                present=list(spec.labels),
                pack_labels=list(manifest.labels),
                distractor_count=manifest.distractor_count,
                seed=mixture_seed ^ 0x5A5A,
                ensemble=prompt_ensemble,
            )
            outcomes, probe_records = _run_probes(
                adapter,
                mixture_audio,
                mixture_sr,
                probes,
                progress_callback=progress_callback,
                pack=manifest.id,
                condition=condition_name,
                mixture_name=spec.name,
            )
            per_condition_outcomes.setdefault(condition_name, []).extend(outcomes)
            all_outcomes.extend(outcomes)
            per_mixture_records.append(
                {
                    "pack": manifest.id,
                    "condition": condition_name,
                    "mixture_name": spec.name,
                    "spec": spec.to_dict(),
                    "components_present": list(spec.labels),
                    "sources": source_ids,
                    "probes": probe_records,
                }
            )
            _emit_progress(
                progress_callback,
                "mixture_done",
                pack=manifest.id,
                condition=condition_name,
                mixture_name=spec.name,
                mixture_index=mixture_index,
                total_mixtures=total_mixtures,
            )

        per_condition = {
            condition: aggregate(items) for condition, items in per_condition_outcomes.items()
        }
        totals = aggregate(all_outcomes)
        pack_results[manifest.id] = {
            "manifest": _serializable_manifest(manifest),
            "per_condition": per_condition,
            "per_class": per_class_breakdown(all_outcomes),
            "totals": totals,
        }

    if custom_outcomes:
        pack_id = next(iter(pack_results)) if pack_results else runtimes[0].manifest.id
        per_condition = pack_results[pack_id].setdefault("per_condition", {})
        per_condition[CUSTOM_CONDITION] = aggregate(custom_outcomes)
        if skip_default_mixtures:
            pack_results[pack_id]["totals"] = aggregate(custom_outcomes)
            pack_results[pack_id]["per_class"] = per_class_breakdown(custom_outcomes)

    pack_summaries: dict[str, dict] = {}
    components_present_total = 0.0
    components_understood_total = 0.0
    weighted_recall_num = 0.0
    weighted_recall_den = 0.0
    weighted_fpr_num = 0.0
    weighted_fpr_den = 0.0
    for pack_id, data in pack_results.items():
        totals = data["totals"]
        components_present_total += totals.get("components_present", 0.0)
        components_understood_total += totals.get("components_understood", 0.0)
        weighted_recall_num += totals.get("components_understood", 0.0)
        weighted_recall_den += totals.get("components_present", 0.0)
        negatives = totals.get("fp", 0.0) + totals.get("tn", 0.0)
        weighted_fpr_num += totals.get("fp", 0.0)
        weighted_fpr_den += negatives
        pack_summaries[pack_id] = {
            "title": data["manifest"]["title"],
            "license_tag": data["manifest"]["license_tag"],
            "totals": totals,
            "per_condition": data["per_condition"],
            "per_class": data["per_class"],
            "scope_note": data["manifest"]["scope_note"],
        }

    weighted_recall = weighted_recall_num / weighted_recall_den if weighted_recall_den else 0.0
    weighted_fpr = weighted_fpr_num / weighted_fpr_den if weighted_fpr_den else 0.0

    cliff = None
    cliff_packs = [data["per_condition"] for data in pack_results.values() if data["per_condition"]]
    if cliff_packs:
        solo_vals = [pc.get("solo", {}).get("recall") for pc in cliff_packs if pc.get("solo")]
        quad_vals = [pc.get("quad", {}).get("recall") for pc in cliff_packs if pc.get("quad")]
        if solo_vals and quad_vals:
            cliff = float(sum(quad_vals) / len(quad_vals)) - float(sum(solo_vals) / len(solo_vals))

    paraphrases_used = list(prompt_spec.paraphrases[: prompt_ensemble or 1])
    config = {
        "model": adapter.name,
        "seed": rng_seed,
        "packs": list(pack_results.keys()),
        "skipped_packs": skipped,
        "profile": profile.name if profile else None,
        "custom_mixtures": [spec.to_dict() for spec in (custom_mixtures or [])],
        "selected_conditions": selected_conditions or [name for name, _ in CONDITIONS],
        "limit": limit,
        "prompt_version": prompt_spec.version,
        "parser_version": prompt_spec.parser_version,
        "prompt_ensemble": prompt_ensemble,
        "prompt_source": prompt_spec.source,
        "prompt_paraphrases_used": paraphrases_used,
        "prompt_paraphrases_hash": prompt_spec.paraphrases_hash,
    }

    digest_run = run_hash(
        suite=SUITE_ID,
        revision=SUITE_REVISION,
        manifest_digest=sha256_text(stable_json([data["manifest"] for data in pack_results.values()])),
        config=config,
        hypotheses=per_mixture_records,
    )
    _emit_progress(progress_callback, "done", run_hash=digest_run)

    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "model": adapter.name,
        "seed": rng_seed,
        "profile": profile.name if profile else None,
        "packs": list(pack_results.keys()),
        "skipped_packs": [{"pack": pid, "reason": reason} for pid, reason in skipped],
        "conditions": [name for name, _ in CONDITIONS] + ([CUSTOM_CONDITION] if custom_outcomes else []),
        "pack_summaries": pack_summaries,
        "headline": {
            "components_understood": int(components_understood_total),
            "components_present": int(components_present_total),
            "weighted_recall": weighted_recall,
            "weighted_fpr": weighted_fpr,
            "solo_quad_cliff": cliff,
        },
        "per_mixture": per_mixture_records,
        "prompt_version": prompt_spec.version,
        "parser_version": prompt_spec.parser_version,
        "prompt_ensemble": prompt_ensemble,
        "prompt_source": prompt_spec.source,
        "prompt_paraphrases_used": paraphrases_used,
        "config": config,
        "run_hash": digest_run,
    }


def _run_probes(
    adapter: AudioLLMAdapter,
    audio: np.ndarray,
    sample_rate: int,
    probes: list[Probe],
    *,
    progress_callback: ProgressCallback | None = None,
    pack: str | None = None,
    condition: str | None = None,
    mixture_name: str | None = None,
) -> tuple[list[ProbeOutcome], list[dict]]:
    outcomes: list[ProbeOutcome] = []
    records: list[dict] = []
    for probe in probes:
        paraphrase_results: list[dict] = []
        per_paraphrase_yes: list[bool] = []
        for prompt_index, prompt in enumerate(probe.prompts, start=1):
            _emit_progress(
                progress_callback,
                "probe_start",
                pack=pack,
                condition=condition,
                mixture_name=mixture_name,
                label=probe.label,
                prompt_index=prompt_index,
                prompt_total=len(probe.prompts),
            )
            reset_adapter_progress = _set_adapter_progress_callback(
                adapter,
                progress_callback,
                suite=SUITE_ID,
                pack=pack,
                condition=condition,
                mixture_name=mixture_name,
                label=probe.label,
                prompt=prompt,
                prompt_index=prompt_index,
                prompt_total=len(probe.prompts),
            )
            try:
                raw = adapter.answer(audio, sample_rate, prompt)
            finally:
                reset_adapter_progress()
            yes = parse_yes_no(raw)
            _emit_progress(
                progress_callback,
                "probe_done",
                suite=SUITE_ID,
                pack=pack,
                condition=condition,
                mixture_name=mixture_name,
                label=probe.label,
                expected=probe.expected,
                prompt=prompt,
                raw_answer=raw,
                answered_yes=yes,
                prompt_index=prompt_index,
                prompt_total=len(probe.prompts),
            )
            per_paraphrase_yes.append(yes)
            paraphrase_results.append(
                {"prompt": prompt, "raw_answer": raw, "answered_yes": yes}
            )
        if len(per_paraphrase_yes) == 1:
            answered_yes = per_paraphrase_yes[0]
        else:
            answered_yes = majority_vote(per_paraphrase_yes)
        outcomes.append(
            ProbeOutcome(label=probe.label, expected=probe.expected, answered_yes=answered_yes)
        )
        records.append(
            {
                "label": probe.label,
                "prompt": probe.primary_prompt,
                "expected": probe.expected,
                "raw_answer": paraphrase_results[0]["raw_answer"],
                "answered_yes": answered_yes,
                "paraphrase_answers": paraphrase_results,
            }
        )
    return outcomes, records


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"event": event, **payload})


def _set_adapter_progress_callback(
    adapter: AudioLLMAdapter,
    callback: ProgressCallback | None,
    **context: Any,
) -> Callable[[], None]:
    setter = getattr(adapter, "set_progress_callback", None)
    if callback is None or setter is None:
        return lambda: None

    def emit(event: dict[str, Any]) -> None:
        callback({**context, **event})

    setter(emit)
    return lambda: setter(None)


def _serializable_manifest(manifest: PackManifest) -> dict:
    return {
        "id": manifest.id,
        "title": manifest.title,
        "source": manifest.source,
        "license": manifest.license,
        "license_tag": manifest.license_tag,
        "scope_note": manifest.scope_note,
        "labels": list(manifest.labels),
        "clip_resolver": manifest.clip_resolver,
        "cache_subdir": manifest.cache_subdir,
        "expected_layout": manifest.expected_layout,
        "mixture_counts": manifest.mixture_counts,
        "distractor_count": manifest.distractor_count,
    }
