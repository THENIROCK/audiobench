from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_text(stable_json(manifest))


def run_hash(
    *,
    suite: str,
    revision: str,
    manifest_digest: str,
    config: dict[str, Any],
    hypotheses: list[dict[str, Any]],
) -> str:
    payload = {
        "suite": suite,
        "revision": revision,
        "manifest_hash": manifest_digest,
        "config": config,
        "hypotheses": hypotheses,
    }
    return sha256_text(stable_json(payload))
