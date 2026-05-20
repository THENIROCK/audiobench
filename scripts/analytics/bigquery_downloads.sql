-- BigQuery: audiobench install profile over the last @days days.
-- Dataset is free to scan up to the BigQuery free tier (~1 TB/month). This
-- query filters by project first so a single run typically scans < 100 MB.
--
-- Required parameters (set with --parameter on `bq query` or in the Python
-- client): @days INT64.
--
-- Returns one row per (country, installer, python_minor) combo with download
-- counts for pip-initiated installs only. Mirrors and CI-bot patterns are
-- filtered out to bias toward "real" users.

WITH base AS (
  SELECT
    timestamp,
    country_code,
    file.version AS version,
    details.installer.name AS installer,
    details.python AS python_full,
    REGEXP_EXTRACT(details.python, r'^(\d+\.\d+)') AS python_minor,
    details.system.name AS os,
    details.distro.name AS distro,
    details.cpu AS cpu
  FROM `bigquery-public-data.pypi.file_downloads`
  WHERE file.project = 'audiobench'
    AND DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
    AND details.installer.name IN ('pip', 'poetry', 'uv', 'pdm', 'hatch')
)
SELECT
  country_code,
  installer,
  python_minor,
  os,
  COUNT(*) AS downloads,
  COUNT(DISTINCT version) AS distinct_versions
FROM base
GROUP BY country_code, installer, python_minor, os
ORDER BY downloads DESC;
