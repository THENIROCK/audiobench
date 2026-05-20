from __future__ import annotations

import json
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

from audiobench.findings import detect_hallucination_findings
from audiobench.data.asr_hallucination.procedural import synthesize_clip
from audiobench.hashing import manifest_hash, run_hash
from audiobench.metrics import compute_asr_signal_metrics
from audiobench.models.asr import ASRAdapter, asr_result_to_dict, normalize_asr_response
from audiobench.models.asr_registry import make_model


SUITE_ID = "ab/asr-hallucination"
SUITE_REVISION = "0.1.0"


def _manifest_path() -> Path:
    return files("audiobench.data.asr_hallucination").joinpath("manifest.json")


def load_manifest() -> dict:
    with _manifest_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _condition_order(manifest: dict, observed: set[str]) -> list[str]:
    ordered = [name for name in manifest.get("conditions", []) if name in observed]
    for name in sorted(observed):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _slice_metadata(clip: dict) -> dict:
    return {
        "duration_s": float(clip.get("duration_s", 0.0)),
        "domain": str(clip.get("domain", "unknown")),
        "noise_type": clip.get("noise_type"),
        "snr_db": clip.get("snr_db"),
        "language": clip.get("language"),
        "accent": clip.get("accent"),
        "slice_tags": list(clip.get("slice_tags", [])),
    }


def run_suite(
    *,
    model_name: str,
    seed: int,
    limit: int | None = None,
    condition_names: list[str] | None = None,
    model: ASRAdapter | None = None,
) -> dict:
    manifest = load_manifest()
    clips = list(manifest.get("clips", []))
    if condition_names:
        allowed = set(condition_names)
        clips = [clip for clip in clips if str(clip.get("domain", "")) in allowed]
    if limit:
        clips = clips[:limit]
    if not clips:
        raise ValueError("no clips selected")

    sample_rate = int(manifest.get("sample_rate", 16000))
    transcriber: ASRAdapter = model if model is not None else make_model(model_name, seed=seed)

    all_refs: list[str] = []
    all_hyps: list[str] = []
    all_non_speech: list[bool] = []
    observed_conditions: set[str] = set()

    refs_by_condition: dict[str, list[str]] = defaultdict(list)
    hyps_by_condition: dict[str, list[str]] = defaultdict(list)
    non_speech_by_condition: dict[str, list[bool]] = defaultdict(list)
    latency_by_condition: dict[str, list[float]] = defaultdict(list)
    cost_by_condition: dict[str, list[float]] = defaultdict(list)
    errors_by_condition: dict[str, int] = defaultdict(int)

    per_clip_results: list[dict] = []
    per_clip_hypotheses: list[dict] = []

    for clip in clips:
        metadata = _slice_metadata(clip)
        domain = str(metadata["domain"])
        observed_conditions.add(domain)

        audio = synthesize_clip(clip, sample_rate=sample_rate, global_seed=seed)
        response = transcriber.transcribe(audio, sample_rate)
        result = normalize_asr_response(response)
        payload = asr_result_to_dict(result)
        transcript = result.transcript
        reference = str(clip.get("reference", ""))

        all_refs.append(reference)
        all_hyps.append(transcript)
        all_non_speech.append(True)

        refs_by_condition[domain].append(reference)
        hyps_by_condition[domain].append(transcript)
        non_speech_by_condition[domain].append(True)
        if result.latency_ms is not None:
            latency_by_condition[domain].append(result.latency_ms)
        if result.cost_usd is not None:
            cost_by_condition[domain].append(result.cost_usd)
        if result.error:
            errors_by_condition[domain] += 1

        per_clip_results.append(
            {
                "clip_id": clip.get("clip_id", clip.get("id")),
                "id": int(clip["id"]),
                "generator": clip.get("generator"),
                "metadata": metadata,
                "reference": reference,
                "result": payload,
            }
        )
        per_clip_hypotheses.append(
            {
                "clip_id": clip.get("clip_id", clip.get("id")),
                "reference": reference,
                "hypotheses": {domain: transcript},
                "metadata": metadata,
                "condition_details": {domain: payload},
            }
        )

    conditions = _condition_order(manifest, observed_conditions)
    per_condition_metrics: dict[str, dict] = {}
    per_condition_runtime: dict[str, dict] = {}
    for condition in conditions:
        condition_refs = refs_by_condition[condition]
        condition_hyps = hyps_by_condition[condition]
        condition_mask = non_speech_by_condition[condition]
        per_condition_metrics[condition] = compute_asr_signal_metrics(
            condition_refs,
            condition_hyps,
            non_speech_mask=condition_mask,
        )
        latencies = latency_by_condition[condition]
        costs = cost_by_condition[condition]
        total_rows = len(condition_refs) or 1
        per_condition_runtime[condition] = {
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
            "error_rate": errors_by_condition[condition] / float(total_rows),
        }

    headline = compute_asr_signal_metrics(all_refs, all_hyps, non_speech_mask=all_non_speech)
    non_speech_total = sum(
        int(per_condition_metrics[condition].get("non_speech_count", 0)) for condition in conditions
    )
    hallucination_weighted_num = sum(
        float(per_condition_metrics[condition]["non_speech_hallucination_rate"])
        * float(per_condition_metrics[condition]["non_speech_count"])
        for condition in conditions
    )
    weighted_hallucination_rate = (
        hallucination_weighted_num / float(non_speech_total) if non_speech_total else 0.0
    )
    findings_bundle = detect_hallucination_findings(per_clip_results, seed=seed)

    config = {
        "model": model_name,
        "seed": seed,
        "clip_count": len(clips),
        "conditions": conditions,
        "condition_key": "domain",
        "sample_rate": sample_rate,
    }
    digest = manifest_hash(manifest)
    digest_run = run_hash(
        suite=SUITE_ID,
        revision=SUITE_REVISION,
        manifest_digest=digest,
        config=config,
        hypotheses=per_clip_results,
    )
    return {
        "suite": SUITE_ID,
        "revision": SUITE_REVISION,
        "model": transcriber.name,
        "seed": seed,
        "clip_count": len(clips),
        "conditions": conditions,
        "condition_key": "domain",
        "manifest_hash": digest,
        "headline": headline,
        "hallucination_rate": headline["non_speech_hallucination_rate"],
        "weighted_hallucination_rate": weighted_hallucination_rate,
        "per_condition_metrics": per_condition_metrics,
        "per_condition_runtime": per_condition_runtime,
        "per_clip_hypotheses": per_clip_hypotheses,
        "per_clip_results": per_clip_results,
        "findings": findings_bundle["findings"],
        "top_finding": findings_bundle["top_finding"],
        "validation_summary": findings_bundle["validation_summary"],
        "findings_methods": findings_bundle["methods"],
        "run_hash": digest_run,
    }
