from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

# Gradio 4.44 + Starlette 1.0 breaks Jinja template caching; keep localhost off proxies on Spaces.
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
_no_proxy = "localhost,127.0.0.1,0.0.0.0,::1"
os.environ.setdefault("NO_PROXY", _no_proxy)
os.environ.setdefault("no_proxy", _no_proxy)

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_DATASET_REPO = os.getenv(
    "AUDIOBENCH_LEADERBOARD_DATASET", "THENIROCK/audiobench-leaderboard-submissions"
)
DEFAULT_TOKEN = os.getenv("HF_TOKEN")
SUBMISSION_PREFIX = "submissions/"
TABLE_COLUMNS = [
    "rank",
    "suite",
    "model",
    "authored_by",
    "score_metric",
    "score",
    "secondary_metric",
    "secondary_score",
    "finding_status",
    "validated_findings",
    "top_finding",
    "top_effect",
    "top_q",
    "revision",
    "run_hash",
    "submitted_by",
    "submitted_at",
    "notes",
    "tags",
    "dataset_file",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _submission_paths(api: HfApi, repo_id: str) -> list[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(
        [path for path in files if path.startswith(SUBMISSION_PREFIX) and path.endswith(".json")]
    )


def _read_submission(api: HfApi, repo_id: str, path: str) -> dict[str, Any]:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path,
        token=DEFAULT_TOKEN,
    )
    with open(local_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_from_submission(submission: dict[str, Any], path: str) -> dict[str, Any]:
    leaderboard = submission.get("leaderboard") or {}
    metrics = leaderboard.get("metrics") or {}
    tags = submission.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    row = {
        "suite": submission.get("suite"),
        "model": submission.get("model"),
        "authored_by": submission.get("authored_by"),
        "score_metric": leaderboard.get("primary_metric"),
        "score": _float(leaderboard.get("primary_value")),
        "secondary_metric": leaderboard.get("secondary_metric"),
        "secondary_score": (
            _float(leaderboard.get("secondary_value"))
            if leaderboard.get("secondary_value") is not None
            else None
        ),
        "finding_status": metrics.get("top_finding_status"),
        "validated_findings": int(metrics.get("validated_findings", 0) or 0),
        "top_finding": metrics.get("top_finding_title") or metrics.get("top_finding_id"),
        "top_effect": (
            _float(metrics.get("top_finding_effect_size"))
            if metrics.get("top_finding_effect_size") is not None
            else None
        ),
        "top_q": (
            _float(metrics.get("top_finding_adjusted_p_value"))
            if metrics.get("top_finding_adjusted_p_value") is not None
            else None
        ),
        "higher_is_better": bool(leaderboard.get("higher_is_better", True)),
        "revision": submission.get("revision"),
        "run_hash": submission.get("run_hash"),
        "submitted_by": submission.get("submitted_by"),
        "submitted_at": submission.get("submitted_at"),
        "notes": submission.get("notes"),
        "tags": ", ".join(str(item) for item in tags),
        "dataset_file": path,
    }
    return row


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    suites = sorted({row["suite"] for row in rows if row.get("suite")})
    for suite in suites:
        suite_rows = [row for row in rows if row.get("suite") == suite]
        suite_rows.sort(
            key=lambda row: (
                -row["score"] if row.get("higher_is_better") else row["score"],
                row["secondary_score"] if row.get("secondary_score") is not None else float("inf"),
                -_timestamp(row.get("submitted_at")).timestamp(),
            )
        )
        for index, row in enumerate(suite_rows, start=1):
            row["rank"] = index


def _to_table_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    rows.sort(
        key=lambda row: (
            row.get("suite") or "",
            int(row.get("rank") or 10_000),
            -_timestamp(row.get("submitted_at")).timestamp(),
        )
    )
    table: list[list[Any]] = []
    for row in rows:
        table.append([row.get(column) for column in TABLE_COLUMNS])
    return table


def load_leaderboard(repo_id: str, suite_filter: str, max_rows: int) -> tuple[list[list[Any]], str]:
    repo_id = repo_id.strip()
    if not repo_id:
        return [], "Set a dataset repo id first."
    api = HfApi(token=DEFAULT_TOKEN)
    try:
        submission_paths = _submission_paths(api, repo_id)
    except Exception as exc:
        return [], f"Could not list files from `{repo_id}`: `{exc}`"

    if not submission_paths:
        return [], f"No submissions found in `{repo_id}` yet."

    submissions: list[dict[str, Any]] = []
    parse_errors = 0
    for path in submission_paths:
        if len(submissions) >= max_rows:
            break
        try:
            payload = _read_submission(api, repo_id, path)
            submissions.append(_row_from_submission(payload, path))
        except Exception:
            parse_errors += 1

    _rank_rows(submissions)
    rows = submissions
    if suite_filter != "all":
        rows = [row for row in submissions if row.get("suite") == suite_filter]
    table = _to_table_rows(rows)
    summary = (
        f"Loaded {len(rows)} rows from `{repo_id}` "
        f"({len(submissions)} parsed, {parse_errors} skipped)."
    )
    return table, summary


with gr.Blocks(title="audiobench leaderboard") as demo:
    gr.Markdown(
        "# audiobench leaderboard\n"
        "Upload with `audiobench push` and this Space will rank submissions automatically."
    )
    with gr.Row():
        repo_input = gr.Textbox(
            label="Leaderboard dataset repo",
            value=DEFAULT_DATASET_REPO,
            placeholder="org-or-user/audiobench-leaderboard",
        )
        suite_filter = gr.Dropdown(
            label="Suite",
            choices=["all", "ab/sound-id", "ab/asr-robust", "ab/asr-hallucination"],
            value="all",
        )
        max_rows = gr.Slider(
            minimum=10,
            maximum=5000,
            value=1000,
            step=10,
            label="Max rows to load",
        )
    refresh = gr.Button("Refresh leaderboard", variant="primary")
    summary = gr.Markdown()
    leaderboard = gr.Dataframe(
        headers=TABLE_COLUMNS,
        interactive=False,
        wrap=True,
    )

    refresh.click(
        fn=load_leaderboard,
        inputs=[repo_input, suite_filter, max_rows],
        outputs=[leaderboard, summary],
    )
    demo.load(
        fn=load_leaderboard,
        inputs=[repo_input, suite_filter, max_rows],
        outputs=[leaderboard, summary],
    )


def _space_port() -> int:
    raw = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860"
    try:
        return int(raw)
    except ValueError:
        return 7860


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=_space_port(),
        show_error=True,
    )
