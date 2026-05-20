"""Pull Google Search Console queries (requires OAuth service account setup)."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def pull(
    site_url: str | None = None,
    credentials_path: str | None = None,
    days: int = 28,
) -> dict[str, Any]:
    site = site_url or os.environ.get("GSC_SITE_URL", "sc-domain:audiobench.dev")
    creds = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds or not Path(creds).exists():
        return {
            "_skipped": True,
            "_reason": "GOOGLE_APPLICATION_CREDENTIALS not set or file missing",
        }

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return {
            "_skipped": True,
            "_reason": "pip install google-api-python-client google-auth",
        }

    scopes = ["https://www.googleapis.com/auth/webmasters.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        creds, scopes=scopes
    )
    service = build("searchconsole", "v1", credentials=credentials, cache_discovery=False)

    end = date.today() - timedelta(days=3)
    start = end - timedelta(days=days)

    body = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dimensions": ["query", "page", "country"],
        "rowLimit": 100,
    }
    try:
        resp = (
            service.searchanalytics()
            .query(siteUrl=site, body=body)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc), "site": site}

    rows = resp.get("rows") or []
    by_query: dict[str, int] = {}
    by_page: dict[str, int] = {}
    by_country: dict[str, int] = {}
    for row in rows:
        keys = row.get("keys") or []
        clicks = int(row.get("clicks", 0))
        if len(keys) >= 1:
            by_query[keys[0]] = by_query.get(keys[0], 0) + clicks
        if len(keys) >= 2:
            by_page[keys[1]] = by_page.get(keys[1], 0) + clicks
        if len(keys) >= 3:
            by_country[keys[2]] = by_country.get(keys[2], 0) + clicks

    return {
        "site": site,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_clicks": sum(by_query.values()),
        "by_query": sorted(by_query.items(), key=lambda kv: -kv[1])[:20],
        "by_page": sorted(by_page.items(), key=lambda kv: -kv[1])[:15],
        "by_country": sorted(by_country.items(), key=lambda kv: -kv[1])[:15],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analytics/snapshots/gsc/latest.json"),
    )
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    payload = pull(days=args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload.get("_skipped"):
        print(f"skipped: {payload['_reason']}")
    elif payload.get("_error"):
        print(f"error: {payload['_error']}")
    else:
        print(f"wrote {args.out} ({payload.get('total_clicks', 0)} clicks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
