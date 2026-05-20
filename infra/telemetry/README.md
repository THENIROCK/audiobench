# audiobench telemetry (Cloudflare Worker + D1)

Anonymous opt-in CLI usage stats. No IPs stored (only `cf-ipcountry`), no file paths, no audio.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/event` | Ingest one event (204 on success) |
| `GET` | `/v1/summary` | Public aggregates, cached 5 min |

Default URL (set in CLI): `https://audiobench-telemetry.thenirock.workers.dev/v1/event`

Override locally: `export AUDIOBENCH_TELEMETRY_URL=https://your-worker.dev/v1/event`

## Deploy

```bash
cd infra/telemetry
npm install -g wrangler   # or: npx wrangler

# Create D1 database
wrangler d1 create audiobench-telemetry
# Copy database_id into wrangler.toml (REPLACE_WITH_D1_DATABASE_ID)

wrangler d1 execute audiobench-telemetry --remote --file=schema.sql

wrangler deploy
```

Free tier: 100k Worker requests/day, 5M D1 rows read/day — more than enough for early adoption.

## Event schema

```json
{
  "install_id": "uuid",
  "event": "cmd.run",
  "version": "0.1.2",
  "py_minor": "3.12",
  "os": "Darwin",
  "ts": 1710000000,
  "suite": "ab/sound-id",
  "adapter": "heuristic-v0",
  "duration_ms": 4200,
  "ok": 1
}
```

Retention: 90 days (`RETENTION_DAYS` in wrangler.toml). Pruned on ~1% of writes.

## Privacy

- IP addresses are never written to D1.
- `install_id` is a random UUID stored only in `~/.config/audiobench/consent.json`.
- `/v1/summary` returns counts only — no install_ids.
