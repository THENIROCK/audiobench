from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import soundfile as sf

from audiobench.hashing import manifest_hash, run_hash
from audiobench.metrics import compute_asr_signal_metrics, compute_wer
from audiobench.models.asr import ASRAdapter, asr_result_to_dict, normalize_asr_response
from audiobench.models.asr_registry import make_model
from audiobench.perturbations import bandlimited_8k, noise_cafe_10db, noise_pink_5db, reverb_medium


SUITE_ID = "ab/asr-robust"
SUITE_REVISION = "0.1.0"


@dataclass(frozen=True)
class Condition:
    name: str
    transform: Callable[[object, int, int], tuple[object, int]]


CONDITIONS: list[Condition] = [
    Condition("clean", lambda audio, sr, seed: (audio, sr)),
    Condition("noise-cafe-10db", noise_cafe_10db),
    Condition("noise-pink-5db", noise_pink_5db),
    Condition("bandlimited-8k", bandlimited_8k),
    Condition("reverb-medium", reverb_medium),
]
ProgressCallback = Callable[[dict[str, Any]], None]


def _manifest_path() -> Path:
    return files("audiobench.data.asr_robust").joinpath("manifest.json")


def load_manifest() -> dict:
    with _manifest_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_suite(
    *,
    model_name: str,
    seed: int,
    limit: int | None = None,
    condition_names: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    model: ASRAdapter | None = None,
) -> dict:
    manifest = load_manifest()
    clips = manifest["clips"][:limit] if limit else manifest["clips"]
    clip_count = len(clips)

    selected_conditions = CONDITIONS
    if condition_names:
        allowed = set(condition_names)
        selected_conditions = [item for item in CONDITIONS if item.name in allowed]
    if not selected_conditions:
        raise ValueError("no conditions selected")

    transcriber: ASRAdapter = model if model is not None else make_model(model_name, seed=seed)
    condition_refs: dict[str, list[str]] = {item.name: [] for item in selected_conditions}
    condition_hyps: dict[str, list[str]] = {item.name: [] for item in selected_conditions}
    condition_latencies: dict[str, list[float]] = {item.name: [] for item in selected_conditions}
    condition_costs: dict[str, list[float]] = {item.name: [] for item in selected_conditions}
    condition_error_counts: dict[str, int] = {item.name: 0 for item in selected_conditions}
    per_clip_hypotheses: list[dict] = []

    data_dir = _manifest_path().parent / "clips"
    total_steps = clip_count * len(selected_conditions)
    _emit_progress(
        progress_callback,
        "start",
        suite=SUITE_ID,
        model=model_name,
        clips=clip_count,
        conditions=[item.name for item in selected_conditions],
        total_steps=total_steps,
    )
    for clip_index, clip in enumerate(clips, start=1):
        clip_path = data_dir / clip["file"]
        audio, sample_rate = sf.read(clip_path)
        ref = clip["text"]
        condition_outputs: dict[str, str] = {}
        condition_details: dict[str, dict] = {}
        for condition in selected_conditions:
            _emit_progress(
                progress_callback,
                "condition_start",
                clip_id=clip["id"],
                clip_index=clip_index,
                clip_total=clip_count,
                condition=condition.name,
            )
            condition_seed = int(manifest["condition_seeds"][condition.name]) + int(clip["id"])
            perturbed, perturbed_sr = condition.transform(audio, int(sample_rate), condition_seed)
            response = transcriber.transcribe(perturbed, perturbed_sr)
            normalized = normalize_asr_response(response)
            hyp = normalized.transcript
            condition_refs[condition.name].append(ref)
            condition_hyps[condition.name].append(hyp)
            condition_outputs[condition.name] = hyp
            condition_details[condition.name] = asr_result_to_dict(normalized)
            if normalized.latency_ms is not None:
                condition_latencies[condition.name].append(normalized.latency_ms)
            if normalized.cost_usd is not None:
                condition_costs[condition.name].append(normalized.cost_usd)
            if normalized.error:
                condition_error_counts[condition.name] += 1
            _emit_progress(
                progress_callback,
                "condition_done",
                clip_id=clip["id"],
                clip_index=clip_index,
                clip_total=clip_count,
                condition=condition.name,
            )
        per_clip_hypotheses.append(
            {
                "clip_id": clip["id"],
                "file": clip["file"],
                "reference": ref,
                "hypotheses": condition_outputs,
                "condition_details": condition_details,
            }
        )

    per_condition_wer = {
        condition.name: compute_wer(condition_refs[condition.name], condition_hyps[condition.name])
        for condition in selected_conditions
    }
    per_condition_runtime = {}
    for condition in selected_conditions:
        latencies = condition_latencies[condition.name]
        costs = condition_costs[condition.name]
        total_rows = len(condition_refs[condition.name]) or 1
        per_condition_runtime[condition.name] = {
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
            "error_rate": condition_error_counts[condition.name] / float(total_rows),
        }
    all_refs = [ref for condition in selected_conditions for ref in condition_refs[condition.name]]
    all_hyps = [hyp for condition in selected_conditions for hyp in condition_hyps[condition.name]]
    asr_signal_metrics = compute_asr_signal_metrics(all_refs, all_hyps)
    weighted_mean_wer = sum(per_condition_wer.values()) / len(per_condition_wer)

    config = {
        "model": model_name,
        "seed": seed,
        "clip_count": clip_count,
        "conditions": [item.name for item in selected_conditions],
    }
    digest = manifest_hash(manifest)
    digest_run = run_hash(
        suite=SUITE_ID,
        revision=SUITE_REVISION,
        manifest_digest=digest,
        config=config,
        hypotheses=per_clip_hypotheses,
    )
    _emit_progress(progress_callback, "done", run_hash=digest_run)
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "model": transcriber.name,
        "seed": seed,
        "clip_count": clip_count,
        "conditions": [item.name for item in selected_conditions],
        "manifest_hash": digest,
        "per_condition_wer": per_condition_wer,
        "weighted_mean_wer": weighted_mean_wer,
        "per_condition_runtime": per_condition_runtime,
        "asr_signal_metrics": asr_signal_metrics,
        "per_clip_hypotheses": per_clip_hypotheses,
        "run_hash": digest_run,
    }


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"event": event, **payload})
