# audiobench analytics

Lightweight tooling for understanding who installs and reads about audiobench,
without collecting personal data.

PyPI exposes no demographic data and no install referrer. The richest signals
we can legitimately collect are:

| Source                  | What it gives us                                          | How          |
|-------------------------|-----------------------------------------------------------|--------------|
| `pypistats.org` API     | Downloads by Python version, OS, mirrors (last ~180 days) | Public HTTP  |
| BigQuery `pypi`         | Per-download rows: country, installer, distro, CPU        | Free quota   |
| GoatCounter (docs site) | Page views, country, referrer for `docs/`                 | JS pixel     |
| GitHub Traffic API      | Top referrers and paths (last 14 days)                    | `gh` CLI     |
| HF Space analytics      | Space views and unique visitors                           | HF dashboard |

This folder covers the first two. The docs-site analytics are wired into
`mkdocs.yml`. GitHub Traffic is a one-liner shown at the bottom.

## Quickstart

```bash
pip install pypistats
python scripts/analytics/pypistats_snapshot.py
```

Writes one JSON and one CSV per day under `analytics/snapshots/pypistats/`.

For the richer BigQuery cut:

```bash
pip install google-cloud-bigquery pandas
gcloud auth application-default login
python scripts/analytics/bigquery_snapshot.py --days 30
```

Writes `analytics/snapshots/bigquery/YYYY-MM-DD-by-country.csv` and
`...-by-installer.csv`.

## Automated weekly snapshot

`.github/workflows/analytics.yml` runs `pypistats_snapshot.py` every Monday at
07:00 UTC and commits the new files. No secrets required. To also run the
BigQuery snapshot in CI, add a `GCP_SA_KEY` secret with a service account that
has `bigquery.jobUser` and `bigquery.dataViewer`; the workflow is preconfigured
to skip the BQ step when the secret is absent.

## GitHub Traffic (referrers)

GitHub only retains 14 days, so snapshot weekly:

```bash
gh api repos/THENIROCK/audiobench/traffic/popular/referrers \
  > analytics/snapshots/github/$(date +%F)-referrers.json
gh api repos/THENIROCK/audiobench/traffic/popular/paths \
  > analytics/snapshots/github/$(date +%F)-paths.json
gh api repos/THENIROCK/audiobench/traffic/views \
  > analytics/snapshots/github/$(date +%F)-views.json
```

The weekly workflow runs these too when `GH_TRAFFIC_TOKEN` (a PAT with
`repo` scope) is set as a repo secret.

## What we deliberately do NOT collect

- No telemetry from the `audiobench` CLI. No phone-home, no run ids, no IPs.
- No cookies on the docs site (GoatCounter is cookieless).
- No personally identifying data in any snapshot committed to this repo.

If that ever changes it will be opt-in, documented on the docs index page, and
shipped behind a flag.
