# Analytics setup

How to wire optional data sources into the [public analytics dashboard](../analytics/index.html).

## Already automatic

These run weekly via [.github/workflows/analytics.yml](https://github.com/THENIROCK/audiobench/blob/main/.github/workflows/analytics.yml) and on every docs deploy:

| Source | Script | Auth |
|--------|--------|------|
| PyPI downloads | `pypistats_snapshot.py` | None |
| GitHub Traffic | `gh api` | `GITHUB_TOKEN` |
| Public mentions | `mentions.py` | None |
| Dashboard HTML | `dashboard.py --site` | None |

## BigQuery (PyPI country breakdown)

Gives per-country install counts (pip/poetry/uv only, mirrors filtered).

1. Create a GCP project and enable BigQuery.
2. `gcloud auth application-default login`
3. Run once locally:

```bash
pip install google-cloud-bigquery
python scripts/analytics/bigquery_snapshot.py --days 30
```

4. For CI: create a service account with `bigquery.jobUser` + `bigquery.dataViewer`, add JSON key as repo secret `GCP_SA_KEY`.

Snapshots land in `analytics/snapshots/bigquery/`.

## GoatCounter (docs referrers + countries)

The docs site loads GoatCounter from [docs/javascripts/analytics.js](../javascripts/analytics.js). Subdomain is `audiobench`.

1. Sign up at [goatcounter.com](https://www.goatcounter.com/).
2. Create API token (Settings → API).
3. Add repo secret `GOATCOUNTER_API_TOKEN` (and optionally `GOATCOUNTER_SITE=audiobench`).

```bash
export GOATCOUNTER_API_TOKEN=...
python scripts/analytics/goatcounter_pull.py
```

## Google Search Console (search queries)

Shows which Google queries drive traffic to `audiobench.dev`.

1. Verify the property in [Search Console](https://search.google.com/search-console).
2. Create a service account, add it as a user on the property.
3. Download JSON key → `GOOGLE_APPLICATION_CREDENTIALS=/path/key.json`
4. Set `GSC_SITE_URL` (e.g. `sc-domain:audiobench.dev` or `https://thenirock.github.io/audiobench/`).

```bash
pip install google-api-python-client google-auth
python scripts/analytics/gsc_pull.py
```

Verification can take 24–48 hours before data appears.

## CLI telemetry worker (opt-in usage)

Deploy the Cloudflare Worker in [infra/telemetry/](https://github.com/THENIROCK/audiobench/tree/main/infra/telemetry):

```bash
cd infra/telemetry
wrangler d1 create audiobench-telemetry
# paste database_id into wrangler.toml
wrangler d1 execute audiobench-telemetry --remote --file=schema.sql
wrangler deploy
```

Set `AUDIOBENCH_TELEMETRY_URL` in releases if you use a custom hostname. The dashboard reads `AUDIOBENCH_TELEMETRY_SUMMARY_URL` (defaults to `/v1/summary` on the same host).

## Refresh the published dashboard

```bash
python scripts/analytics/mentions.py
python scripts/analytics/dashboard.py --site
mkdocs gh-deploy --clean
```

Or wait for the Monday cron / the `docs` workflow on push to `main`.
