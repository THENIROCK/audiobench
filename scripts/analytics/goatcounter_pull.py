"""Pull aggregate stats from GoatCounter API (docs site referrers + countries)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

USER_AGENT = "audiobench-analytics/1.0"


def pull(
    site_code: str | None = None,
    api_token: str | None = None,
    days: int = 30,
) -> dict[str, Any]:
    site = site_code or os.environ.get("GOATCOUNTER_SITE", "audiobench")
    token = api_token or os.environ.get("GOATCOUNTER_API_TOKEN", "")
    if not token:
        return {
            "_skipped": True,
            "_reason": "GOATCOUNTER_API_TOKEN not set",
        }

    base = f"https://{site}.goatcounter.com/api/v0/stats"
    end = date.today().isoformat()
    start = date.today().replace(day=1).isoformat()  # fallback; API uses range param

    def _get(path: str, extra: dict[str, str] | None = None) -> Any:
        params = {"start": f"{days}d", "end": end}
        if extra:
            params.update(extra)
        url = f"{base}/{path}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return {"_error": f"HTTP {exc.code}", "_url": url}
        except urllib.error.URLError as exc:
            return {"_error": str(exc.reason), "_url": url}

    total = _get("total")
    refs = _get("refs")
    countries = _get("locations")

    by_referrer: list[tuple[str, int]] = []
    if isinstance(refs, dict) and "refs" in refs:
        for row in refs.get("refs") or []:
            name = row.get("ref") or row.get("name") or "direct"
            by_referrer.append((name, int(row.get("count", 0))))
        by_referrer.sort(key=lambda kv: -kv[1])

    by_country: list[tuple[str, int]] = []
    if isinstance(countries, dict) and "locations" in countries:
        for row in countries.get("locations") or []:
            code = row.get("code") or row.get("name") or "??"
            by_country.append((code, int(row.get("count", 0))))
        by_country.sort(key=lambda kv: -kv[1])

    pageviews = 0
    if isinstance(total, dict):
        pageviews = int((total.get("stats") or {}).get("count", 0) or total.get("count", 0) or 0)

    return {
        "site": site,
        "days": days,
        "pageviews": pageviews,
        "by_referrer": by_referrer[:15],
        "by_country": by_country[:20],
        "raw": {"total": total, "refs": refs, "locations": countries},
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analytics/snapshots/goatcounter/latest.json"),
    )
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    payload = pull(days=args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload.get("_skipped"):
        print(f"skipped: {payload['_reason']}")
    else:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
