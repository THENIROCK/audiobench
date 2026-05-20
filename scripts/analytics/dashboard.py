"""Pull everything we can about audiobench right now and render one HTML page.

Sources:
- pypistats.org      (recent / overall / python_minor / system, last ~180 days)
- GitHub Traffic API (views, clones, referrers, paths, last 14 days) via `gh`
- GitHub repo API    (stars, forks, watchers, open issues, age) via `gh`
- HF Hub API         (Space + dataset + model download/like counts) via HTTP

Outputs:
- analytics/dashboard-data.json   (raw merged blob, useful for audits)
- analytics/dashboard.html        (self-contained, opens in any browser)

Usage:
    python scripts/analytics/dashboard.py
    python scripts/analytics/dashboard.py --open

No GCP / no secrets / no network calls beyond the three public APIs above.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PYPI_PROJECT = "audiobench"
GH_REPO = "THENIROCK/audiobench"
HF_TARGETS: list[tuple[str, str]] = [
]
HF_AUTHOR = "THENIROCK"
USER_AGENT = "audiobench-analytics/1.0"


def _http_json(url: str, attempts: int = 4) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504) and i < attempts:
                time.sleep(2**i)
                continue
            return {"_error": f"HTTP {exc.code}", "_url": url}
        except urllib.error.URLError as exc:
            last = exc
            if i < attempts:
                time.sleep(2**i)
                continue
            return {"_error": str(exc.reason), "_url": url}
    return {"_error": str(last), "_url": url}


def _gh(path: str) -> Any:
    try:
        out = subprocess.check_output(
            ["gh", "api", path], stderr=subprocess.STDOUT, text=True
        )
        return json.loads(out)
    except FileNotFoundError:
        return {"_error": "gh CLI not installed"}
    except subprocess.CalledProcessError as exc:
        return {"_error": exc.output.strip()[:500]}


def collect_pypistats() -> dict[str, Any]:
    base = f"https://pypistats.org/api/packages/{PYPI_PROJECT}"
    endpoints = ("recent", "overall", "python_minor", "system")
    out: dict[str, Any] = {}
    for i, e in enumerate(endpoints):
        if i:
            time.sleep(1.5)
        out[e] = _http_json(f"{base}/{e}")
    return out


def collect_github() -> dict[str, Any]:
    return {
        "repo": _gh(f"repos/{GH_REPO}"),
        "views": _gh(f"repos/{GH_REPO}/traffic/views"),
        "clones": _gh(f"repos/{GH_REPO}/traffic/clones"),
        "referrers": _gh(f"repos/{GH_REPO}/traffic/popular/referrers"),
        "paths": _gh(f"repos/{GH_REPO}/traffic/popular/paths"),
    }


def collect_hf() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    targets = list(HF_TARGETS)
    if HF_AUTHOR and not targets:
        for kind in ("spaces", "models", "datasets"):
            listing = _http_json(f"https://huggingface.co/api/{kind}?author={HF_AUTHOR}")
            if isinstance(listing, list):
                for item in listing:
                    repo_id = item.get("id") or item.get("modelId")
                    if repo_id:
                        targets.append((kind.rstrip("s"), repo_id))
    for kind, repo_id in targets:
        if kind == "space":
            url = f"https://huggingface.co/api/spaces/{repo_id}"
        elif kind == "dataset":
            url = f"https://huggingface.co/api/datasets/{repo_id}"
        else:
            url = f"https://huggingface.co/api/models/{repo_id}"
        results.append({"kind": kind, "id": repo_id, "data": _http_json(url)})
    return results


def aggregate(raw: dict[str, Any]) -> dict[str, Any]:
    pys = raw["pypistats"]
    gh = raw["github"]

    recent = (pys.get("recent") or {}).get("data") or {}
    overall = (pys.get("overall") or {}).get("data") or []
    overall = [r for r in overall if isinstance(r, dict)]
    daily_total: dict[str, int] = {}
    for row in overall:
        if row.get("category") == "with_mirrors":
            continue
        d = row.get("date")
        if d:
            daily_total[d] = daily_total.get(d, 0) + int(row.get("downloads", 0))
    daily_sorted = sorted(daily_total.items())

    def _normalise_cat(value: Any) -> str:
        if value in (None, "null", "", "Other"):
            return "Unknown / mirrors"
        return str(value)

    py_minor_data = ((pys.get("python_minor") or {}).get("data") or [])
    by_python: dict[str, int] = {}
    for row in py_minor_data:
        cat = _normalise_cat(row.get("category"))
        by_python[cat] = by_python.get(cat, 0) + int(row.get("downloads", 0))

    os_data = ((pys.get("system") or {}).get("data") or [])
    by_os: dict[str, int] = {}
    for row in os_data:
        cat = _normalise_cat(row.get("category"))
        by_os[cat] = by_os.get(cat, 0) + int(row.get("downloads", 0))

    total_downloads_all_time = sum(daily_total.values())

    repo = gh.get("repo") or {}
    views = gh.get("views") or {}
    clones = gh.get("clones") or {}
    referrers = gh.get("referrers") or []
    paths = gh.get("paths") or []

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": {
            "pypi": PYPI_PROJECT,
            "github": GH_REPO,
            "created_at": repo.get("created_at"),
        },
        "kpis": {
            "downloads_last_day": recent.get("last_day"),
            "downloads_last_week": recent.get("last_week"),
            "downloads_last_month": recent.get("last_month"),
            "downloads_total_180d": total_downloads_all_time,
            "github_stars": repo.get("stargazers_count"),
            "github_forks": repo.get("forks_count"),
            "github_watchers": repo.get("subscribers_count"),
            "github_open_issues": repo.get("open_issues_count"),
            "views_14d": views.get("count"),
            "unique_visitors_14d": views.get("uniques"),
            "clones_14d": clones.get("count"),
            "unique_cloners_14d": clones.get("uniques"),
        },
        "series": {
            "downloads_daily": [
                {"date": d, "downloads": n} for d, n in daily_sorted
            ],
            "views_daily": views.get("views") or [],
            "clones_daily": clones.get("clones") or [],
        },
        "breakdowns": {
            "by_python_minor": sorted(
                by_python.items(), key=lambda kv: -kv[1]
            ),
            "by_os": sorted(by_os.items(), key=lambda kv: -kv[1]),
            "referrers": referrers,
            "paths": paths,
        },
        "hf": raw["hf"],
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>audiobench analytics</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
  .kpi-num { font-variant-numeric: tabular-nums; }
  .chart-box { position: relative; width: 100%; }
  .chart-box > canvas { position: absolute; inset: 0; width: 100% !important; height: 100% !important; }
</style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
<div class="max-w-7xl mx-auto px-6 py-10">
  <header class="flex items-end justify-between mb-10">
    <div>
      <h1 class="text-3xl font-semibold tracking-tight">audiobench</h1>
      <p class="text-slate-400 mt-1">PyPI + GitHub + Hugging Face, in one view.</p>
    </div>
    <div class="text-right text-xs text-slate-500">
      <div>Generated <span id="generated_at"></span></div>
      <div>Sources: pypistats.org / GitHub Traffic API / HF Hub API</div>
    </div>
  </header>

  <section class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8" id="kpis"></section>

  <section class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
    <div class="card rounded-xl p-5 lg:col-span-3">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">PyPI downloads, last 180 days</h2>
      <div class="chart-box" style="height:280px;"><canvas id="chart_downloads"></canvas></div>
    </div>
  </section>

  <section class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Downloads by Python version</h2>
      <div class="chart-box" style="height:280px;"><canvas id="chart_python"></canvas></div>
    </div>
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Downloads by OS</h2>
      <div class="chart-box" style="height:280px;"><canvas id="chart_os"></canvas></div>
    </div>
  </section>

  <section class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">GitHub page views, last 14 days</h2>
      <div class="chart-box" style="height:240px;"><canvas id="chart_views"></canvas></div>
    </div>
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">GitHub clones, last 14 days</h2>
      <div class="chart-box" style="height:240px;"><canvas id="chart_clones"></canvas></div>
    </div>
  </section>

  <section class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Top referrers (GitHub)</h2>
      <table class="w-full text-sm" id="tbl_referrers"></table>
    </div>
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Top repo paths (GitHub)</h2>
      <table class="w-full text-sm" id="tbl_paths"></table>
    </div>
  </section>

  <section class="card rounded-xl p-5 mb-12">
    <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Hugging Face</h2>
    <div id="hf"></div>
  </section>

  <footer class="text-center text-xs text-slate-500 pb-8">
    No personal data collected. PyPI provides no referrer or user data; this is the maximum we can know.
  </footer>
</div>

<script>
const DATA = __DATA__;

document.getElementById("generated_at").textContent = DATA.generated_at + " UTC";

const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());

const kpiDefs = [
  ["Downloads last month",  DATA.kpis.downloads_last_month,   "PyPI / last 30d"],
  ["Downloads last week",   DATA.kpis.downloads_last_week,    "PyPI / last 7d"],
  ["Downloads (180d total)",DATA.kpis.downloads_total_180d,   "PyPI / pypistats overall"],
  ["Downloads yesterday",   DATA.kpis.downloads_last_day,     "PyPI / last 1d"],
  ["GitHub stars",          DATA.kpis.github_stars,           "github.com/" + DATA.project.github],
  ["Unique visitors (14d)", DATA.kpis.unique_visitors_14d,    "GitHub Traffic"],
  ["Page views (14d)",      DATA.kpis.views_14d,              "GitHub Traffic"],
  ["Unique cloners (14d)",  DATA.kpis.unique_cloners_14d,     "GitHub Traffic"],
];
document.getElementById("kpis").innerHTML = kpiDefs.map(([label, val, sub]) => `
  <div class="card rounded-xl p-5">
    <div class="text-xs uppercase tracking-wider text-slate-400">${label}</div>
    <div class="kpi-num text-3xl font-semibold mt-2">${fmt(val)}</div>
    <div class="text-xs text-slate-500 mt-1">${sub}</div>
  </div>
`).join("");

Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "rgba(148,163,184,0.15)";
const ACCENT = "rgba(99,102,241,1)";
const ACCENT_FILL = "rgba(99,102,241,0.18)";

function lineChart(id, labels, values, label) {
  new Chart(document.getElementById(id), {
    type: "line",
    data: { labels, datasets: [{
      label, data: values,
      borderColor: ACCENT, backgroundColor: ACCENT_FILL,
      borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0,
    }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: true, grid: { color: "rgba(148,163,184,0.08)" } },
      },
    },
  });
}

function donutChart(id, labels, values) {
  new Chart(document.getElementById(id), {
    type: "doughnut",
    data: { labels, datasets: [{
      data: values,
      backgroundColor: ["#6366f1","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#64748b","#0ea5e9","#14b8a6"],
      borderWidth: 0,
    }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
      cutout: "62%",
    },
  });
}

lineChart(
  "chart_downloads",
  DATA.series.downloads_daily.map(r => r.date),
  DATA.series.downloads_daily.map(r => r.downloads),
  "downloads",
);
lineChart(
  "chart_views",
  (DATA.series.views_daily || []).map(r => r.timestamp.slice(0,10)),
  (DATA.series.views_daily || []).map(r => r.count),
  "views",
);
lineChart(
  "chart_clones",
  (DATA.series.clones_daily || []).map(r => r.timestamp.slice(0,10)),
  (DATA.series.clones_daily || []).map(r => r.count),
  "clones",
);

donutChart(
  "chart_python",
  DATA.breakdowns.by_python_minor.map(r => r[0]),
  DATA.breakdowns.by_python_minor.map(r => r[1]),
);
donutChart(
  "chart_os",
  DATA.breakdowns.by_os.map(r => r[0]),
  DATA.breakdowns.by_os.map(r => r[1]),
);

function table(id, rows, cols) {
  const head = `<thead><tr class="text-left text-slate-400 text-xs uppercase tracking-wider">${cols.map(c => `<th class="py-2">${c.label}</th>`).join("")}</tr></thead>`;
  const body = rows.length
    ? rows.map(r => `<tr class="border-t border-slate-800">${cols.map(c => `<td class="py-2">${c.fmt ? c.fmt(r[c.key]) : (r[c.key] ?? "—")}</td>`).join("")}</tr>`).join("")
    : `<tr><td colspan="${cols.length}" class="py-4 text-center text-slate-500">No data yet.</td></tr>`;
  document.getElementById(id).innerHTML = head + "<tbody>" + body + "</tbody>";
}

table("tbl_referrers", DATA.breakdowns.referrers || [], [
  { key: "referrer", label: "Source" },
  { key: "count",    label: "Views", fmt: fmt },
  { key: "uniques",  label: "Unique", fmt: fmt },
]);
table("tbl_paths", (DATA.breakdowns.paths || []).slice(0, 10), [
  { key: "path",    label: "Path" },
  { key: "count",   label: "Views", fmt: fmt },
  { key: "uniques", label: "Unique", fmt: fmt },
]);

const hfRoot = document.getElementById("hf");
if (!DATA.hf || !DATA.hf.length) {
  hfRoot.innerHTML = `<div class="text-slate-500 text-sm">No HF targets configured.</div>`;
} else {
  hfRoot.innerHTML = DATA.hf.map(item => {
    const d = item.data || {};
    if (d._error) return `<div class="text-sm text-slate-500">${item.kind} <code>${item.id}</code>: ${d._error}</div>`;
    return `
      <div class="flex items-center justify-between border-t border-slate-800 first:border-0 py-3">
        <div>
          <div class="text-sm font-medium">${item.kind} · ${item.id}</div>
          <div class="text-xs text-slate-500">${d.sdk || d.pipeline_tag || ""} ${d.private ? "(private)" : ""}</div>
        </div>
        <div class="flex gap-6 text-right">
          <div><div class="text-xs text-slate-400">Likes</div><div class="kpi-num text-lg">${fmt(d.likes)}</div></div>
          <div><div class="text-xs text-slate-400">Downloads</div><div class="kpi-num text-lg">${fmt(d.downloads)}</div></div>
        </div>
      </div>`;
  }).join("");
}
</script>
</body>
</html>
"""


def render(merged: dict[str, Any], out_html: Path) -> None:
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(merged))
    out_html.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("analytics"),
        help="Directory for the rendered dashboard.",
    )
    parser.add_argument(
        "--filename", default="dashboard.html",
        help="HTML filename. Use index.html when writing into a docs site.",
    )
    parser.add_argument(
        "--site", action="store_true",
        help="Shortcut: write to docs/analytics/index.html (mkdocs picks it up).",
    )
    parser.add_argument(
        "--no-data-dump", action="store_true",
        help="Skip writing dashboard-data.json (useful for the published site).",
    )
    parser.add_argument("--open", action="store_true", help="Open the dashboard in your browser.")
    args = parser.parse_args()

    if args.site:
        args.out_dir = Path("docs/analytics")
        args.filename = "index.html"
        args.no_data_dump = True

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("pulling pypistats ...")
    pypistats = collect_pypistats()
    print("pulling github ...")
    github = collect_github()
    print("pulling hugging face ...")
    hf = collect_hf()

    raw = {"pypistats": pypistats, "github": github, "hf": hf}
    merged = aggregate(raw)

    html_path = args.out_dir / args.filename
    if not args.no_data_dump:
        data_path = args.out_dir / "dashboard-data.json"
        data_path.write_text(
            json.dumps({"raw": raw, "merged": merged}, indent=2), encoding="utf-8"
        )
    render(merged, html_path)

    k = merged["kpis"]
    print()
    print(f"  downloads last month : {k['downloads_last_month']}")
    print(f"  downloads last week  : {k['downloads_last_week']}")
    print(f"  github stars         : {k['github_stars']}")
    print(f"  unique visitors 14d  : {k['unique_visitors_14d']}")
    print()
    if not args.no_data_dump:
        print(f"  wrote {args.out_dir / 'dashboard-data.json'}")
    print(f"  wrote {html_path}")

    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
