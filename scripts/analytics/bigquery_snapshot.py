"""Snapshot BigQuery pypi download stats for audiobench.

Runs ``bigquery_downloads.sql`` and writes the result to
``analytics/snapshots/bigquery/<date>-by-country.csv`` plus a small
``summary.json`` aggregated across country / installer / python_minor.

Requires:
    pip install google-cloud-bigquery
    gcloud auth application-default login
or, in CI, GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account key
with ``bigquery.jobUser`` and ``bigquery.dataViewer``.

Usage:
    python scripts/analytics/bigquery_snapshot.py --days 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

SQL_PATH = Path(__file__).with_name("bigquery_downloads.sql")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days.")
    parser.add_argument("--out", default="analytics/snapshots/bigquery", type=Path)
    parser.add_argument(
        "--project",
        default=None,
        help="GCP billing project. Defaults to gcloud's default project.",
    )
    args = parser.parse_args()

    try:
        from google.cloud import bigquery
    except ImportError:
        print(
            "google-cloud-bigquery is not installed. Run `pip install google-cloud-bigquery`.",
            file=sys.stderr,
        )
        return 2

    client = bigquery.Client(project=args.project)
    sql = SQL_PATH.read_text(encoding="utf-8")
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", args.days)],
    )

    print(f"running BigQuery (lookback={args.days}d) ...")
    rows = list(client.query(sql, job_config=job_config).result())
    if not rows:
        print("no rows returned (project may have zero downloads in window)")
        return 0

    today = date.today().isoformat()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{today}-by-country-installer-python.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row.items()))

    by_country: dict[str, int] = defaultdict(int)
    by_installer: dict[str, int] = defaultdict(int)
    by_python: dict[str, int] = defaultdict(int)
    total = 0
    for row in rows:
        downloads = int(row["downloads"])
        total += downloads
        by_country[row["country_code"] or "??"] += downloads
        by_installer[row["installer"] or "??"] += downloads
        by_python[row["python_minor"] or "??"] += downloads

    summary = {
        "snapshot_date": today,
        "lookback_days": args.days,
        "total_downloads": total,
        "by_country": dict(sorted(by_country.items(), key=lambda kv: -kv[1])),
        "by_installer": dict(sorted(by_installer.items(), key=lambda kv: -kv[1])),
        "by_python_minor": dict(sorted(by_python.items(), key=lambda kv: -kv[1])),
    }
    (out_dir / f"{today}-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"wrote {csv_path}")
    print(f"total downloads in window: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
