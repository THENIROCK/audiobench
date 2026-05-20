"""Collect public mentions of audiobench across HN, Reddit, Bluesky, GitHub, libraries.io."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

QUERY = "audiobench"
USER_AGENT = "audiobench-analytics/1.0"


def _http_json(url: str, headers: dict[str, str] | None = None) -> Any:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"_error": f"HTTP {exc.code}", "_url": url}
    except urllib.error.URLError as exc:
        return {"_error": str(exc.reason), "_url": url}


def _norm(
    source: str,
    *,
    url: str,
    title: str,
    author: str = "",
    ts: str = "",
    snippet: str = "",
) -> dict[str, str]:
    return {
        "source": source,
        "url": url,
        "title": title[:200],
        "author": author[:80],
        "ts": ts,
        "snippet": snippet[:300],
    }


def collect_hn() -> list[dict[str, str]]:
    data = _http_json(
        "https://hn.algolia.com/api/v1/search?"
        + urllib.parse.urlencode({"query": QUERY, "tags": "story,comment", "hitsPerPage": 50})
    )
    if not isinstance(data, dict) or "hits" not in data:
        return []
    out: list[dict[str, str]] = []
    for hit in data.get("hits") or []:
        obj_id = hit.get("objectID", "")
        story_id = hit.get("story_id") or obj_id
        is_comment = hit.get("_tags") and "comment" in (hit.get("_tags") or [])
        if is_comment:
            link = f"https://news.ycombinator.com/item?id={obj_id}"
        else:
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
        ts = ""
        if hit.get("created_at_i"):
            ts = datetime.fromtimestamp(
                int(hit["created_at_i"]), tz=timezone.utc
            ).isoformat()
        out.append(
            _norm(
                "hn",
                url=link,
                title=hit.get("title") or hit.get("comment_text", "")[:200] or "(HN item)",
                author=hit.get("author") or "",
                ts=ts,
                snippet=hit.get("story_text") or hit.get("comment_text") or "",
            )
        )
    return out


def collect_reddit() -> list[dict[str, str]]:
    data = _http_json(
        "https://www.reddit.com/search.json?"
        + urllib.parse.urlencode({"q": QUERY, "sort": "new", "limit": "50"}),
        headers={"User-Agent": "audiobench-analytics:1.0 (contact: github.com/THENIROCK/audiobench)"},
    )
    if not isinstance(data, dict):
        return []
    out: list[dict[str, str]] = []
    for child in (data.get("data") or {}).get("children") or []:
        post = (child or {}).get("data") or {}
        out.append(
            _norm(
                "reddit",
                url="https://www.reddit.com" + (post.get("permalink") or ""),
                title=post.get("title") or "(reddit post)",
                author=post.get("author") or "",
                ts=datetime.fromtimestamp(
                    float(post.get("created_utc") or 0), tz=timezone.utc
                ).isoformat()
                if post.get("created_utc")
                else "",
                snippet=post.get("selftext") or "",
            )
        )
    return out


def collect_bluesky() -> list[dict[str, str]]:
    data = _http_json(
        "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?"
        + urllib.parse.urlencode({"q": QUERY, "limit": "50"})
    )
    if not isinstance(data, dict):
        return []
    out: list[dict[str, str]] = []
    for post in data.get("posts") or []:
        record = post.get("record") or {}
        handle = (post.get("author") or {}).get("handle") or ""
        uri = post.get("uri") or ""
        rkey = uri.split("/")[-1] if uri else ""
        link = f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else ""
        out.append(
            _norm(
                "bluesky",
                url=link,
                title=(record.get("text") or "")[:200] or "(bluesky post)",
                author=handle,
                ts=record.get("createdAt") or "",
                snippet=record.get("text") or "",
            )
        )
    return out


def collect_github_issues() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                "gh",
                "search",
                "issues",
                QUERY,
                "--json",
                "url,title,repository,createdAt,author",
                "--limit",
                "30",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        items = json.loads(out)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []
    results: list[dict[str, str]] = []
    for item in items:
        repo = item.get("repository") or {}
        repo_name = repo.get("nameWithOwner") or repo.get("name") or ""
        author = (item.get("author") or {}).get("login") or ""
        results.append(
            _norm(
                "github",
                url=item.get("url") or "",
                title=item.get("title") or "(issue)",
                author=author,
                ts=item.get("createdAt") or "",
                snippet=repo_name,
            )
        )
    return results


def collect_libraries_io() -> list[dict[str, str]]:
    api_key = os.environ.get("LIBRARIES_IO_API_KEY", "")
    url = f"https://libraries.io/api/pypi/{QUERY}/dependents"
    if api_key:
        url += "?" + urllib.parse.urlencode({"api_key": api_key})
    data = _http_json(url)
    if isinstance(data, dict) and data.get("_error"):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for dep in data[:30]:
        name = dep.get("name") or dep.get("title") or "unknown"
        platform = dep.get("platform") or "pypi"
        out.append(
            _norm(
                "libraries.io",
                url=dep.get("package_url") or dep.get("repository_url") or "",
                title=f"{platform}:{name}",
                author=dep.get("owner_name") or "",
                ts=dep.get("latest_release_published_at") or "",
                snippet=f"rank {dep.get('rank', '?')}",
            )
        )
    return out


def collect_all() -> dict[str, Any]:
    print("  mentions: hn ...")
    hn = collect_hn()
    time.sleep(1)
    print("  mentions: reddit ...")
    reddit = collect_reddit()
    time.sleep(1)
    print("  mentions: bluesky ...")
    bsky = collect_bluesky()
    time.sleep(0.5)
    print("  mentions: github ...")
    gh = collect_github_issues()
    print("  mentions: libraries.io ...")
    deps = collect_libraries_io()

    all_items = hn + reddit + bsky + gh + deps
    all_items.sort(key=lambda m: m.get("ts") or "", reverse=True)

    counts = {
        "hn": len(hn),
        "reddit": len(reddit),
        "bluesky": len(bsky),
        "github": len(gh),
        "libraries.io": len(deps),
        "total": len(all_items),
    }
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": QUERY,
        "counts": counts,
        "items": all_items[:100],
        "feed": all_items[:25],
    }


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analytics/snapshots/mentions/latest.json"),
    )
    args = parser.parse_args()
    payload = collect_all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({payload['counts']['total']} mentions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
