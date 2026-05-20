"""Snapshot pypistats.org data for the audiobench project.

Writes JSON and CSV files under ``analytics/snapshots/pypistats/`` so we have
a longitudinal record (pypistats only retains ~180 days).

Usage:
    python scripts/analytics/pypistats_snapshot.py
    python scripts/analytics/pypistats_snapshot.py --project audiobench --out analytics/snapshots/pypistats
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

API_BASE = "https://pypistats.org/api/packages"
ENDPOINTS = ("recent", "overall", "python_major", "python_minor", "system")
INTER_REQUEST_DELAY_S = 1.5
MAX_ATTEMPTS = 5


def fetch(project: str, endpoint: str) -> dict[str, Any]:
    url = f"{API_BASE}/{project}/{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "audiobench-analytics/1.0"})
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS:
                backoff = min(60, 2**attempt)
                print(f"pypistats {exc.code} on {endpoint}, retrying in {backoff}s")
                time.sleep(backoff)
                continue
            raise SystemExit(f"pypistats HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise SystemExit(f"pypistats network error for {url}: {exc.reason}") from exc
    raise SystemExit(f"pypistats exhausted retries for {url}: {last_exc}")


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return
    fieldnames = sorted({key for row in data for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="audiobench")
    parser.add_argument("--out", default="analytics/snapshots/pypistats", type=Path)
    args = parser.parse_args()

    today = date.today().isoformat()
    out_dir: Path = args.out / today
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"project": args.project, "snapshot_date": today, "endpoints": {}}
    for i, endpoint in enumerate(ENDPOINTS):
        if i:
            time.sleep(INTER_REQUEST_DELAY_S)
        payload = fetch(args.project, endpoint)
        (out_dir / f"{endpoint}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(out_dir / f"{endpoint}.csv", payload)
        summary["endpoints"][endpoint] = payload.get("data")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote pypistats snapshot for {args.project} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
