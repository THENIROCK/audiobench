from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from audiobench.hashing import sha256_text
from audiobench.metrics import token_count
from audiobench.statistics import benjamini_hochberg, bootstrap_mean_difference


def _clip_identity(sample: dict[str, Any]) -> str:
    clip_id = sample.get("clip_id")
    if clip_id is not None:
        return str(clip_id)
    if sample.get("id") is not None:
        return str(sample.get("id"))
    return "unknown"


def _sample_domain(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    domain = metadata.get("domain")
    if domain is None:
        return "unknown"
    return str(domain)


def _split_bucket(*, clip_identity: str, seed: int, holdout_fraction: float) -> str:
    token = f"{clip_identity}|{seed}|ab-asr-hallucination-holdout"
    draw = int(sha256_text(token)[:8], 16) / float(0xFFFFFFFF)
    return "holdout" if draw < holdout_fraction else "discovery"


def _hallucination_flag(sample: dict[str, Any]) -> int:
    payload = sample.get("result") or {}
    transcript = ""
    if isinstance(payload, dict):
        transcript = str(payload.get("transcript", ""))
    return 1 if token_count(transcript) > 0 else 0


def _rate(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _reproducibility_checklist(samples: Sequence[dict[str, Any]]) -> dict[str, bool]:
    has_identity = all(_clip_identity(sample) != "unknown" for sample in samples)
    has_domain = all(_sample_domain(sample) != "unknown" for sample in samples)
    has_transcript = all(
        isinstance((sample.get("result") or {}).get("transcript", ""), str) for sample in samples
    )
    return {
        "clip_identity_present": has_identity,
        "domain_metadata_present": has_domain,
        "transcript_present": has_transcript,
    }


def detect_hallucination_findings(
    per_clip_results: Sequence[dict[str, Any]],
    *,
    seed: int,
    holdout_fraction: float = 0.3,
    confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    discovery_alpha: float = 0.05,
    holdout_alpha: float = 0.05,
) -> dict[str, Any]:
    """Detect per-domain hallucination findings with holdout validation."""

    samples = [dict(sample) for sample in per_clip_results]
    if not samples:
        return {
            "findings": [],
            "validation_summary": {
                "status_counts": {"validated": 0, "candidate": 0, "rejected": 0},
                "publishable": False,
                "reproducibility_checklist": _reproducibility_checklist(samples),
            },
            "top_finding": None,
            "methods": {
                "bootstrap_resamples": bootstrap_resamples,
                "confidence": confidence,
                "multiple_testing": "benjamini-hochberg",
                "discovery_alpha": discovery_alpha,
                "holdout_alpha": holdout_alpha,
                "holdout_fraction": holdout_fraction,
            },
        }

    normalized: list[dict[str, Any]] = []
    for sample in samples:
        clip_identity = _clip_identity(sample)
        domain = _sample_domain(sample)
        normalized.append(
            {
                "clip_identity": clip_identity,
                "domain": domain,
                "split": _split_bucket(
                    clip_identity=clip_identity,
                    seed=seed,
                    holdout_fraction=holdout_fraction,
                ),
                "hallucination": _hallucination_flag(sample),
            }
        )

    domains = sorted({row["domain"] for row in normalized})
    findings: list[dict[str, Any]] = []
    p_values: list[float] = []
    reproducibility_checklist = _reproducibility_checklist(samples)
    reproducibility_complete = all(reproducibility_checklist.values())

    for domain in domains:
        discovery_domain = [
            row["hallucination"]
            for row in normalized
            if row["split"] == "discovery" and row["domain"] == domain
        ]
        discovery_other = [
            row["hallucination"]
            for row in normalized
            if row["split"] == "discovery" and row["domain"] != domain
        ]
        holdout_domain = [
            row["hallucination"]
            for row in normalized
            if row["split"] == "holdout" and row["domain"] == domain
        ]
        holdout_other = [
            row["hallucination"]
            for row in normalized
            if row["split"] == "holdout" and row["domain"] != domain
        ]

        discovery_domain_rate = _rate(discovery_domain)
        discovery_other_rate = _rate(discovery_other)
        holdout_domain_rate = _rate(holdout_domain)
        holdout_other_rate = _rate(holdout_other)

        discovery_testable = len(discovery_domain) >= 2 and len(discovery_other) >= 2
        holdout_testable = len(holdout_domain) >= 1 and len(holdout_other) >= 1

        detector_seed = int(sha256_text(f"{seed}|{domain}|discovery")[:8], 16)
        holdout_seed = int(sha256_text(f"{seed}|{domain}|holdout")[:8], 16)

        if discovery_testable:
            discovery_interval = bootstrap_mean_difference(
                discovery_domain,
                discovery_other,
                confidence=confidence,
                resamples=bootstrap_resamples,
                seed=detector_seed,
            )
            p_value = discovery_interval.p_value
            effect_size = discovery_interval.estimate
            ci_lower = discovery_interval.lower
            ci_upper = discovery_interval.upper
        else:
            effect_size = discovery_domain_rate - discovery_other_rate
            ci_lower = effect_size
            ci_upper = effect_size
            p_value = 1.0

        if holdout_testable:
            holdout_interval = bootstrap_mean_difference(
                holdout_domain,
                holdout_other,
                confidence=confidence,
                resamples=bootstrap_resamples,
                seed=holdout_seed,
            )
            holdout_effect = holdout_interval.estimate
            holdout_ci_lower = holdout_interval.lower
            holdout_ci_upper = holdout_interval.upper
            holdout_p_value = holdout_interval.p_value
        else:
            holdout_effect = holdout_domain_rate - holdout_other_rate
            holdout_ci_lower = holdout_effect
            holdout_ci_upper = holdout_effect
            holdout_p_value = 1.0

        finding = {
            "id": f"hallucination-rate:{domain}",
            "title": f"{domain} hallucination uplift",
            "detector": "domain_hallucination_uplift",
            "metric": "non_speech_hallucination_rate",
            "slice": {"domain": domain},
            "effect_size": effect_size,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "adjusted_p_value": 1.0,
            "discovery": {
                "clip_count_domain": len(discovery_domain),
                "clip_count_other": len(discovery_other),
                "domain_rate": discovery_domain_rate,
                "other_rate": discovery_other_rate,
                "testable": discovery_testable,
            },
            "holdout": {
                "clip_count_domain": len(holdout_domain),
                "clip_count_other": len(holdout_other),
                "domain_rate": holdout_domain_rate,
                "other_rate": holdout_other_rate,
                "effect_size": holdout_effect,
                "ci_lower": holdout_ci_lower,
                "ci_upper": holdout_ci_upper,
                "p_value": holdout_p_value,
                "testable": holdout_testable,
            },
            "status": "rejected",
            "validation_notes": "",
            "rank": 0,
        }
        findings.append(finding)
        p_values.append(p_value if discovery_testable else 1.0)

    adjusted = benjamini_hochberg(p_values)
    for finding, q_value in zip(findings, adjusted):
        finding["adjusted_p_value"] = q_value
        effect_size = float(finding["effect_size"])
        ci_lower = float(finding["ci_lower"])
        discovery_testable = bool(finding["discovery"]["testable"])
        holdout_testable = bool(finding["holdout"]["testable"])
        holdout_effect = float(finding["holdout"]["effect_size"])
        holdout_ci_lower = float(finding["holdout"]["ci_lower"])
        holdout_p_value = float(finding["holdout"]["p_value"])

        discovery_pass = (
            discovery_testable
            and effect_size > 0.0
            and ci_lower > 0.0
            and q_value <= discovery_alpha
        )
        holdout_pass = (
            holdout_testable
            and holdout_effect > 0.0
            and holdout_ci_lower > 0.0
            and holdout_p_value <= holdout_alpha
        )

        if not reproducibility_complete:
            finding["status"] = "rejected"
            finding["validation_notes"] = "reproducibility checklist incomplete"
        elif discovery_pass and holdout_pass:
            finding["status"] = "validated"
            finding["validation_notes"] = "passed discovery + holdout gate"
        elif discovery_pass:
            finding["status"] = "candidate"
            finding["validation_notes"] = "passed discovery gate, pending holdout replication"
        else:
            finding["status"] = "rejected"
            if not discovery_testable:
                finding["validation_notes"] = "insufficient discovery support"
            else:
                finding["validation_notes"] = "failed discovery significance gate"

    priority = {"validated": 0, "candidate": 1, "rejected": 2}
    findings.sort(
        key=lambda item: (
            priority.get(str(item["status"]), 3),
            -abs(float(item["effect_size"])),
            float(item["adjusted_p_value"]),
            -(int(item["discovery"]["clip_count_domain"]) + int(item["holdout"]["clip_count_domain"])),
        )
    )
    for index, finding in enumerate(findings, start=1):
        finding["rank"] = index

    status_counts = Counter(str(item["status"]) for item in findings)
    top_finding = findings[0] if findings else None
    return {
        "findings": findings,
        "validation_summary": {
            "status_counts": {
                "validated": int(status_counts.get("validated", 0)),
                "candidate": int(status_counts.get("candidate", 0)),
                "rejected": int(status_counts.get("rejected", 0)),
            },
            "publishable": int(status_counts.get("validated", 0)) > 0 and reproducibility_complete,
            "reproducibility_checklist": reproducibility_checklist,
        },
        "top_finding": top_finding,
        "methods": {
            "bootstrap_resamples": bootstrap_resamples,
            "confidence": confidence,
            "multiple_testing": "benjamini-hochberg",
            "discovery_alpha": discovery_alpha,
            "holdout_alpha": holdout_alpha,
            "holdout_fraction": holdout_fraction,
        },
    }
