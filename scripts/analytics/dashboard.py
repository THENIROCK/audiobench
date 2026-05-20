"""Pull everything we can about audiobench right now and render one HTML page.

Sources:
- pypistats.org, GitHub Traffic, HF Hub
- Public mentions (HN, Reddit, Bluesky, GitHub issues, libraries.io)
- Optional snapshots: BigQuery geo, GoatCounter, GSC, CLI telemetry summary

Usage:
    python scripts/analytics/dashboard.py
    python scripts/analytics/dashboard.py --site --open
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
HF_TARGETS: list[tuple[str, str]] = []
HF_AUTHOR = "THENIROCK"
USER_AGENT = "audiobench-analytics/1.0"
SNAPSHOTS_DIR = Path("analytics/snapshots")
TELEMETRY_SUMMARY_URL = os.environ.get(
    "AUDIOBENCH_TELEMETRY_SUMMARY_URL",
    "https://audiobench-telemetry.thenirock.workers.dev/v1/summary",
)


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_file(glob_pattern: str) -> Path | None:
    matches = sorted(SNAPSHOTS_DIR.glob(glob_pattern))
    return matches[-1] if matches else None


def load_snapshot_mentions() -> dict[str, Any] | None:
    path = SNAPSHOTS_DIR / "mentions" / "latest.json"
    if path.exists():
        return _load_json(path)
    return None


def collect_mentions() -> dict[str, Any]:
    cached = load_snapshot_mentions()
    if cached and cached.get("items"):
        return cached
    spec = importlib.util.spec_from_file_location(
        "mentions",
        Path(__file__).with_name("mentions.py"),
    )
    if spec is None or spec.loader is None:
        return {"counts": {}, "items": [], "feed": []}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_all()


def load_bigquery_summary() -> dict[str, Any] | None:
    bq_dir = SNAPSHOTS_DIR / "bigquery"
    if not bq_dir.is_dir():
        return None
    summaries = sorted(bq_dir.glob("*-summary.json"))
    return _load_json(summaries[-1]) if summaries else None


def load_goatcounter() -> dict[str, Any] | None:
    return _load_json(SNAPSHOTS_DIR / "goatcounter" / "latest.json")


def load_gsc() -> dict[str, Any] | None:
    return _load_json(SNAPSHOTS_DIR / "gsc" / "latest.json")


def collect_telemetry_summary() -> dict[str, Any]:
    data = _http_json(TELEMETRY_SUMMARY_URL, attempts=2)
    if isinstance(data, dict) and not data.get("_error"):
        return data
    return {"_empty": True, "_reason": data.get("_error", "no data yet")}


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
        "growth": {
            "mentions": raw.get("mentions") or {},
            "goatcounter": raw.get("goatcounter"),
            "gsc": raw.get("gsc"),
            "bigquery": raw.get("bigquery"),
        },
        "telemetry": raw.get("telemetry") or {},
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
      <p class="text-slate-400 mt-1">PyPI, GitHub, mentions, geo, and opt-in CLI usage.</p>
    </div>
    <div class="text-right text-xs text-slate-500">
      <div>Generated <span id="generated_at"></span></div>
      <div>Sources: pypistats / GitHub / HN / Reddit / Bluesky / GoatCounter / CLI telemetry</div>
    </div>
  </header>

  <section class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8" id="kpis"></section>

  <section class="mb-8">
    <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Where from (mentions)</h2>
    <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="growth_kpis"></div>
  </section>

  <section class="card rounded-xl p-5 mb-8">
    <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Mentions feed (latest 25)</h2>
    <table class="w-full text-sm" id="tbl_mentions"></table>
  </section>

  <section class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Geography (PyPI installs)</h2>
      <div class="chart-box" style="height:280px;"><canvas id="chart_geo"></canvas></div>
      <p class="text-xs text-slate-500 mt-2" id="geo_note"></p>
    </div>
    <div class="card rounded-xl p-5">
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Docs site (GoatCounter)</h2>
      <div id="goatcounter_panel" class="text-sm text-slate-400"></div>
    </div>
  </section>

  <section class="card rounded-xl p-5 mb-8">
    <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Real-use telemetry (opt-in CLI)</h2>
    <div id="telemetry_panel"></div>
  </section>

  <section class="card rounded-xl p-5 mb-8" id="gsc_section" style="display:none">
    <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">Search queries (Google Search Console)</h2>
    <table class="w-full text-sm" id="tbl_gsc"></table>
  </section>

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
    CLI telemetry is opt-in only. PyPI provides no install referrer. See docs/reference/telemetry.md.
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

const growth = DATA.growth || {};
const mcounts = (growth.mentions || {}).counts || {};
const growthDefs = [
  ["Hacker News", mcounts.hn || 0, "Algolia search"],
  ["Reddit", mcounts.reddit || 0, "search API"],
  ["Bluesky", mcounts.bluesky || 0, "public API"],
  ["GitHub issues", mcounts.github || 0, "gh search"],
  ["PyPI dependents", mcounts["libraries.io"] || 0, "libraries.io"],
];
document.getElementById("growth_kpis").innerHTML = growthDefs.map(([label, val, sub]) => `
  <div class="card rounded-xl p-5">
    <div class="text-xs uppercase tracking-wider text-slate-400">${label}</div>
    <div class="kpi-num text-3xl font-semibold mt-2">${fmt(val)}</div>
    <div class="text-xs text-slate-500 mt-1">${sub}</div>
  </div>
`).join("");

const feed = (growth.mentions || {}).feed || [];
table("tbl_mentions", feed, [
  { key: "source", label: "Source" },
  { key: "ts", label: "When" },
  { key: "title", label: "Title" },
  { key: "author", label: "Author" },
]);

const bq = growth.bigquery || {};
const byCountry = Object.entries(bq.by_country || {});
const geoNote = document.getElementById("geo_note");
if (!byCountry.length) {
  geoNote.textContent = "No BigQuery geo snapshot yet. Run: python scripts/analytics/bigquery_snapshot.py";
  new Chart(document.getElementById("chart_geo"), {
    type: "bar",
    data: { labels: ["—"], datasets: [{ data: [0], backgroundColor: "#64748b" }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });
} else {
  geoNote.textContent = `Last ${bq.lookback_days || 30}d · ${fmt(bq.total_downloads)} pip installs (filtered)`;
  new Chart(document.getElementById("chart_geo"), {
    type: "bar",
    data: {
      labels: byCountry.slice(0, 12).map(r => r[0]),
      datasets: [{ data: byCountry.slice(0, 12).map(r => r[1]), backgroundColor: "#6366f1" }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    },
  });
}

const gc = growth.goatcounter || {};
const gcPanel = document.getElementById("goatcounter_panel");
if (gc._skipped) {
  gcPanel.innerHTML = `<p>${gc._reason}. See docs/guides/analytics-setup.md</p>`;
} else if (gc._error) {
  gcPanel.innerHTML = `<p class="text-red-400">${gc._error}</p>`;
} else {
  const refs = (gc.by_referrer || []).map(r => `<li>${r[0]}: ${fmt(r[1])}</li>`).join("");
  const countries = (gc.by_country || []).map(r => `<li>${r[0]}: ${fmt(r[1])}</li>`).join("");
  gcPanel.innerHTML = `
    <p class="mb-2">Pageviews (${gc.days || 30}d): <span class="kpi-num text-lg text-white">${fmt(gc.pageviews)}</span></p>
    <div class="grid grid-cols-2 gap-4">
      <div><div class="text-xs uppercase text-slate-500 mb-1">Top referrers</div><ul class="text-xs">${refs || "<li>—</li>"}</ul></div>
      <div><div class="text-xs uppercase text-slate-500 mb-1">Top countries</div><ul class="text-xs">${countries || "<li>—</li>"}</ul></div>
    </div>`;
}

const tel = DATA.telemetry || {};
const telPanel = document.getElementById("telemetry_panel");
if (tel._empty || tel._error) {
  telPanel.innerHTML = `<p class="text-slate-500">No opt-in CLI data yet. Users see a first-run prompt; events appear here after deploy + opt-in.</p>`;
} else {
  const dau = (tel.daily_active || []).slice(-7);
  const err = tel.cmd_stats || {};
  const errRate = err.total ? ((err.failed || 0) / err.total * 100).toFixed(1) : "0";
  telPanel.innerHTML = `
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
      <div><div class="text-xs text-slate-400">DAU (latest day)</div><div class="kpi-num text-2xl">${fmt(dau.length ? dau[dau.length-1].dau : 0)}</div></div>
      <div><div class="text-xs text-slate-400">Events (30d window)</div><div class="kpi-num text-2xl">${fmt((tel.daily_active||[]).reduce((s,r)=>s+(r.events||0),0))}</div></div>
      <div><div class="text-xs text-slate-400">Cmd error rate</div><div class="kpi-num text-2xl">${errRate}%</div></div>
      <div><div class="text-xs text-slate-400">Window</div><div class="kpi-num text-2xl">${tel.window_days || 30}d</div></div>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
      <div><div class="uppercase text-slate-500 mb-1">Top suites</div>${(tel.top_suites||[]).map(r=>`<div>${r.name}: ${r.count}</div>`).join("")||"—"}</div>
      <div><div class="uppercase text-slate-500 mb-1">Top adapters</div>${(tel.top_adapters||[]).map(r=>`<div>${r.name}: ${r.count}</div>`).join("")||"—"}</div>
      <div><div class="uppercase text-slate-500 mb-1">Versions</div>${(tel.top_versions||[]).map(r=>`<div>${r.name}: ${r.count}</div>`).join("")||"—"}</div>
    </div>`;
}

const gsc = growth.gsc || {};
if (gsc.by_query && gsc.by_query.length) {
  document.getElementById("gsc_section").style.display = "block";
  table("tbl_gsc", gsc.by_query.map(([q,c])=>({query:q,clicks:c})), [
    { key: "query", label: "Query" },
    { key: "clicks", label: "Clicks", fmt: fmt },
  ]);
}

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
    print("pulling mentions ...")
    mentions = collect_mentions()
    print("loading snapshots (bigquery / goatcounter / gsc) ...")
    bigquery = load_bigquery_summary()
    goatcounter = load_goatcounter()
    gsc = load_gsc()
    print("pulling telemetry summary ...")
    telemetry = collect_telemetry_summary()

    raw = {
        "pypistats": pypistats,
        "github": github,
        "hf": hf,
        "mentions": mentions,
        "bigquery": bigquery,
        "goatcounter": goatcounter,
        "gsc": gsc,
        "telemetry": telemetry,
    }
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
