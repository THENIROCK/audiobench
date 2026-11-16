(function () {
  "use strict";

  const CACHE_KEY = "ab-leaderboard-files";
  const CACHE_TTL_MS = 5 * 60 * 1000;

  const SUITE_ORDER = ["ab/sound-id", "ab/asr-robust", "ab/asr-hallucination"];

  const SUITE_META = {
    "ab/sound-id": {
      title: "Sound identification",
      hint: "Higher recall is better",
    },
    "ab/asr-robust": {
      title: "ASR robustness",
      hint: "Lower WER is better",
    },
    "ab/asr-hallucination": {
      title: "ASR hallucination",
      hint: "Lower hallucination rate is better",
    },
  };

  function pct(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const n = Number(value);
    const scaled = n <= 1 && n >= 0 ? n * 100 : n;
    return scaled.toFixed(digits ?? 0) + "%";
  }

  function wer(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const n = Number(value);
    const scaled = n <= 1 && n >= 0 ? n * 100 : n;
    return scaled.toFixed(1) + "%";
  }

  function metricLabels(suite, leaderboard) {
    const primary = leaderboard.primary_metric || "";
    const secondary = leaderboard.secondary_metric || "";
    const labels = { primary: primary, secondary: secondary || "" };
    if (suite === "ab/sound-id") {
      labels.primary = "Recall";
      labels.secondary = "FPR";
    } else if (suite === "ab/asr-robust") {
      labels.primary = "Mean WER";
      labels.secondary = secondary ? "Clean WER" : "";
    } else if (suite === "ab/asr-hallucination") {
      labels.primary = "Hallucination";
      labels.secondary = "Non-speech";
    }
    return labels;
  }

  function formatScore(suite, metricName, value) {
    if (value == null) return "—";
    if (suite === "ab/asr-robust") return wer(value);
    if (metricName && metricName.includes("wer")) return wer(value);
    if (metricName && metricName.includes("finding")) {
      return String(value);
    }
    return pct(value, suite === "ab/sound-id" ? 0 : 1);
  }

  function secondaryValue(suite, record) {
    const lb = record.leaderboard || {};
    const metrics = lb.metrics || {};
    if (suite === "ab/asr-hallucination") {
      const runHeadline = (record.run || {}).headline || {};
      if (runHeadline.non_speech_hallucination_rate != null) {
        return {
          value: runHeadline.non_speech_hallucination_rate,
          metric: "non_speech_hallucination_rate",
        };
      }
      if (metrics.top_finding_status) {
        return { value: metrics.top_finding_status, metric: "finding_status" };
      }
    }
    if (lb.secondary_value != null && lb.secondary_metric) {
      return { value: lb.secondary_value, metric: lb.secondary_metric };
    }
    return { value: null, metric: "" };
  }

  function readCache(repo) {
    try {
      const raw = sessionStorage.getItem(CACHE_KEY + ":" + repo);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (Date.now() - parsed.ts > CACHE_TTL_MS) return null;
      return parsed.paths;
    } catch (_e) {
      return null;
    }
  }

  function writeCache(repo, paths) {
    try {
      sessionStorage.setItem(
        CACHE_KEY + ":" + repo,
        JSON.stringify({ ts: Date.now(), paths: paths })
      );
    } catch (_e) {
      /* ignore quota */
    }
  }

  async function listSubmissionPaths(repo) {
    const cached = readCache(repo);
    if (cached) return cached;
    const url =
      "https://huggingface.co/api/datasets/" +
      encodeURIComponent(repo) +
      "/tree/main/submissions?recursive=true";
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not list submissions (" + res.status + ")");
    const entries = await res.json();
    const paths = entries
      .filter(function (e) {
        return e.path && e.path.endsWith(".json");
      })
      .map(function (e) {
        return e.path;
      });
    writeCache(repo, paths);
    return paths;
  }

  async function fetchSubmission(repo, path) {
    const url =
      "https://huggingface.co/datasets/" +
      encodeURIComponent(repo) +
      "/resolve/main/" +
      path;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load " + path);
    return res.json();
  }

  function sortRows(rows) {
    return rows.sort(function (a, b) {
      const ah = a.leaderboard.higher_is_better;
      const av = Number(a.leaderboard.primary_value);
      const bv = Number(b.leaderboard.primary_value);
      if (ah) return bv - av;
      return av - bv;
    });
  }

  function renderCard(suite, rows, limit) {
    const meta = SUITE_META[suite] || { title: suite, hint: "" };
    const article = document.createElement("article");
    article.className = "ab-leaderboard__suite-card";
    article.dataset.suite = suite;

    const header = document.createElement("header");
    header.className = "ab-leaderboard__suite-header";
    header.innerHTML =
      "<h2>" +
      meta.title +
      "</h2><p class=\"ab-leaderboard__suite-hint\">" +
      meta.hint +
      "</p>";
    article.appendChild(header);

    const list = document.createElement("ol");
    list.className = "ab-leaderboard__rows";

    const top = sortRows(rows).slice(0, limit);
    if (top.length === 0) {
      const empty = document.createElement("p");
      empty.className = "ab-leaderboard__empty";
      empty.textContent = "No submissions yet.";
      article.appendChild(empty);
      return article;
    }

    top.forEach(function (record, index) {
      const lb = record.leaderboard || {};
      const labels = metricLabels(suite, lb);
      const sec = secondaryValue(suite, record);
      const li = document.createElement("li");
      li.className = "ab-leaderboard__row";
      if (index === 0) li.classList.add("ab-leaderboard__row--first");

      const rank = document.createElement("span");
      rank.className = "ab-leaderboard__rank";
      rank.textContent = String(index + 1);

      const model = document.createElement("span");
      model.className = "ab-leaderboard__model";
      model.textContent = record.model || "—";

      const scores = document.createElement("span");
      scores.className = "ab-leaderboard__scores";

      const primary = document.createElement("span");
      primary.className = "ab-leaderboard__score ab-leaderboard__score--primary";
      primary.title = labels.primary;
      primary.textContent =
        labels.primary + " " + formatScore(suite, lb.primary_metric, lb.primary_value);

      scores.appendChild(primary);

      if (sec.value != null && labels.secondary) {
        const secondary = document.createElement("span");
        secondary.className = "ab-leaderboard__score ab-leaderboard__score--secondary";
        secondary.title = labels.secondary;
        secondary.textContent =
          labels.secondary + " " + formatScore(suite, sec.metric, sec.value);
        scores.appendChild(secondary);
      }

      li.appendChild(rank);
      li.appendChild(model);
      li.appendChild(scores);
      list.appendChild(li);
    });

    article.appendChild(list);
    return article;
  }

  async function init() {
    const root = document.getElementById("ab-leaderboard");
    if (!root) return;

    const repo = root.dataset.repo || "";
    const author = root.dataset.author || "";
    const limit = parseInt(root.dataset.limit || "5", 10);
    const status = root.querySelector(".ab-leaderboard__status");
    const grid = root.querySelector(".ab-leaderboard__grid");

    if (!repo || !grid) return;

    try {
      const paths = await listSubmissionPaths(repo);
      const records = await Promise.all(
        paths.map(function (p) {
          return fetchSubmission(repo, p);
        })
      );
      const filtered = records.filter(function (r) {
        if (!r || !r.suite) return false;
        if (author && r.authored_by !== author) return false;
        return true;
      });

      const bySuite = {};
      filtered.forEach(function (r) {
        if (!bySuite[r.suite]) bySuite[r.suite] = [];
        bySuite[r.suite].push(r);
      });

      grid.innerHTML = "";
      SUITE_ORDER.forEach(function (suite) {
        grid.appendChild(renderCard(suite, bySuite[suite] || [], limit));
      });

      if (status) {
        status.textContent = "";
        status.hidden = true;
      }
    } catch (err) {
      if (status) {
        status.textContent =
          "Could not load rankings. Check the dataset repo or try again later.";
        status.hidden = false;
      }
      console.error("audiobench leaderboard:", err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
