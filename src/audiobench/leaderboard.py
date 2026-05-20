from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from audiobench.hashing import sha256_text, stable_json


LEADERBOARD_SCHEMA_VERSION = "audiobench.leaderboard.v1"
DEFAULT_DATASET_ENV = "AUDIOBENCH_LEADERBOARD_DATASET"
DEFAULT_SPACE_ENV = "AUDIOBENCH_LEADERBOARD_SPACE"
DEFAULT_DATASET_REPO_BASENAME = "audiobench-leaderboard-submissions"


@dataclass(frozen=True)
class PushResult:
    repo_id: str
    path_in_repo: str
    uploaded: bool
    duplicate: bool
    dataset_url: str
    space_url: str | None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _findings_snapshot(run_payload: dict[str, Any]) -> dict[str, Any]:
    top_finding = run_payload.get("top_finding")
    if not isinstance(top_finding, dict):
        findings = run_payload.get("findings") or []
        if findings and isinstance(findings[0], dict):
            top_finding = findings[0]
        else:
            top_finding = {}
    validation = run_payload.get("validation_summary") or {}
    status_counts = validation.get("status_counts") or {}
    has_top = bool(top_finding.get("id") or top_finding.get("title"))
    return {
        "top_finding_id": top_finding.get("id"),
        "top_finding_title": top_finding.get("title"),
        "top_finding_status": top_finding.get("status"),
        "top_finding_effect_size": (
            _float_or_default(top_finding.get("effect_size"))
            if has_top and top_finding.get("effect_size") is not None
            else None
        ),
        "top_finding_adjusted_p_value": (
            _float_or_default(top_finding.get("adjusted_p_value"), default=1.0)
            if has_top and top_finding.get("adjusted_p_value") is not None
            else None
        ),
        "validated_findings": int(status_counts.get("validated", 0)),
        "candidate_findings": int(status_counts.get("candidate", 0)),
        "rejected_findings": int(status_counts.get("rejected", 0)),
        "publishable_findings": bool(validation.get("publishable", False)),
    }


def _suite_metrics(run_payload: dict[str, Any]) -> dict[str, Any]:
    suite = str(run_payload.get("suite", ""))
    findings = _findings_snapshot(run_payload)
    if suite == "ab/sound-id":
        headline = run_payload.get("headline") or {}
        components_understood = int(headline.get("components_understood", 0))
        components_present = int(headline.get("components_present", 0))
        weighted_recall = _float_or_default(headline.get("weighted_recall"))
        weighted_fpr = _float_or_default(headline.get("weighted_fpr"))
        return {
            "primary_metric": "weighted_recall",
            "primary_value": weighted_recall,
            "higher_is_better": True,
            "secondary_metric": "weighted_fpr",
            "secondary_value": weighted_fpr,
            "metrics": {
                "weighted_recall": weighted_recall,
                "weighted_fpr": weighted_fpr,
                "components_understood": components_understood,
                "components_present": components_present,
                **findings,
            },
        }
    if suite == "ab/asr-robust":
        weighted_mean_wer = _float_or_default(run_payload.get("weighted_mean_wer"))
        per_condition = run_payload.get("per_condition_wer")
        clean_wer = None
        if isinstance(per_condition, dict) and "clean" in per_condition:
            clean_wer = _float_or_default(per_condition.get("clean"))
        metrics: dict[str, Any] = {"weighted_mean_wer": weighted_mean_wer}
        if clean_wer is not None:
            metrics["clean_wer"] = clean_wer
        metrics.update(findings)
        return {
            "primary_metric": "weighted_mean_wer",
            "primary_value": weighted_mean_wer,
            "higher_is_better": False,
            "secondary_metric": "clean_wer" if clean_wer is not None else None,
            "secondary_value": clean_wer,
            "metrics": metrics,
        }
    if suite == "ab/asr-hallucination":
        weighted_hallucination_rate = _float_or_default(
            run_payload.get("weighted_hallucination_rate", run_payload.get("hallucination_rate"))
        )
        top_q = findings.get("top_finding_adjusted_p_value")
        metrics = {
            "weighted_hallucination_rate": weighted_hallucination_rate,
            **findings,
        }
        return {
            "primary_metric": "weighted_hallucination_rate",
            "primary_value": weighted_hallucination_rate,
            "higher_is_better": False,
            "secondary_metric": (
                "top_finding_adjusted_p_value" if findings.get("top_finding_id") is not None else None
            ),
            "secondary_value": top_q if findings.get("top_finding_id") is not None else None,
            "metrics": metrics,
        }
    raise ValueError(
        "unsupported suite "
        f"{suite!r}; leaderboard push currently supports ab/sound-id, "
        "ab/asr-robust, and ab/asr-hallucination"
    )


def submission_path(*, suite: str, run_hash: str) -> str:
    suite_slug = suite.replace("/", "__")
    return f"submissions/{suite_slug}/{run_hash}.json"


def dataset_file_url(*, repo_id: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"


def space_url(space_id: str | None) -> str | None:
    if not space_id:
        return None
    return f"https://huggingface.co/spaces/{space_id}"


def build_submission_record(
    run_payload: dict[str, Any],
    *,
    submitted_by: str | None = None,
    authored_by: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    missing = [key for key in ("suite", "revision", "run_hash", "model") if key not in run_payload]
    if missing:
        raise ValueError(f"run payload missing required keys for push: {', '.join(missing)}")
    suite_metrics = _suite_metrics(run_payload)
    normalized_tags = [item.strip() for item in (tags or []) if item.strip()]
    return {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "suite": run_payload["suite"],
        "revision": run_payload["revision"],
        "model": run_payload["model"],
        "seed": run_payload.get("seed"),
        "run_hash": run_payload["run_hash"],
        "payload_sha256": sha256_text(stable_json(run_payload)),
        "submitted_at": _now_utc_iso(),
        "submitted_by": submitted_by,
        "authored_by": authored_by.strip() if isinstance(authored_by, str) and authored_by.strip() else None,
        "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
        "tags": normalized_tags,
        "source_file": source_file,
        "leaderboard": suite_metrics,
        "run": run_payload,
    }


def _build_hf_api(*, token: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for network push; install dependencies and re-run."
        ) from exc
    return HfApi(token=token) if token else HfApi()


def _whoami(api: Any, *, token: str | None) -> dict[str, Any] | None:
    try:
        return api.whoami(token=token) if token else api.whoami()
    except Exception:
        return None


def _extract_username(identity: dict[str, Any] | None) -> str | None:
    if not isinstance(identity, dict):
        return None
    for field in ("name", "user", "username"):
        value = identity.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def default_dataset_repo_id(*, token: str | None = None, api: Any | None = None) -> str:
    api_client = api if api is not None else _build_hf_api(token=token)
    identity = _whoami(api_client, token=token)
    username = _extract_username(identity)
    if not username:
        raise RuntimeError(
            "could not auto-detect Hugging Face username. "
            "Run `hf auth login`, or pass --repo explicitly."
        )
    return f"{username}/{DEFAULT_DATASET_REPO_BASENAME}"


def _resolve_submitter(api: Any, *, token: str | None) -> str | None:
    identity = _whoami(api, token=token)
    if not isinstance(identity, dict):
        return None
    for field in ("name", "user", "username", "fullname", "email"):
        value = identity.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def push_submission(
    run_payload: dict[str, Any],
    *,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    allow_overwrite: bool = False,
    authored_by: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
    source_file: str | None = None,
    space_id: str | None = None,
    api: Any | None = None,
) -> tuple[dict[str, Any], PushResult]:
    api_client = api if api is not None else _build_hf_api(token=token)
    submitter = _resolve_submitter(api_client, token=token)
    record = build_submission_record(
        run_payload,
        submitted_by=submitter,
        authored_by=authored_by,
        notes=notes,
        tags=tags,
        source_file=source_file,
    )
    path_in_repo = submission_path(suite=record["suite"], run_hash=record["run_hash"])

    try:
        api_client.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        existing_files = set(api_client.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as exc:
        raise RuntimeError(
            f"could not access dataset repo {repo_id!r}. "
            "Make sure your Hugging Face token has write access to that namespace."
        ) from exc
    duplicate = path_in_repo in existing_files

    result = PushResult(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        uploaded=not duplicate or allow_overwrite,
        duplicate=duplicate,
        dataset_url=dataset_file_url(repo_id=repo_id, path_in_repo=path_in_repo),
        space_url=space_url(space_id),
    )
    if duplicate and not allow_overwrite:
        return record, result

    body = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    commit_message = (
        f"audiobench: add {record['suite']} submission "
        f"{record['run_hash'][:8]} ({record['model']})"
    )
    try:
        api_client.upload_file(
            path_or_fileobj=body,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to upload submission to {repo_id!r}: {exc}") from exc
    return record, result
