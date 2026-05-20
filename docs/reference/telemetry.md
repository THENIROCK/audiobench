# Telemetry reference

audiobench collects **nothing by default**. On first interactive use you may be asked to share anonymous CLI usage stats. You can also enable or disable telemetry at any time without reinstalling.

## What is sent (when opted in)

Each command emits at most one event after the command finishes:

| Field | Example | Notes |
|-------|---------|-------|
| `install_id` | UUID | Random id in `~/.config/audiobench/consent.json` — not tied to your identity |
| `event` | `cmd.run` | Command name only |
| `version` | `0.1.2` | audiobench package version |
| `py_minor` | `3.12` | Python minor version |
| `os` | `Linux` | OS family |
| `country` | `US` | From Cloudflare `cf-ipcountry` only — **IP is never stored** |
| `suite` | `ab/sound-id` | When applicable |
| `adapter` | `heuristic-v0` | Model adapter id when applicable |
| `duration_ms` | `4200` | Wall time for the command |
| `ok` | `1` | `1` success, `0` if the command raised |

## What is never sent

- Audio files, transcripts, or run JSON contents
- File paths, hostnames, usernames, or emails
- IP addresses (only coarse country when the event reaches our worker)
- Environment variables or API keys

## Opt in / opt out

```bash
# Force on (skips prompt if consent file missing)
export AUDIOBENCH_TELEMETRY=1

# Force off
export AUDIOBENCH_TELEMETRY=0

# Custom endpoint (self-hosted worker)
export AUDIOBENCH_TELEMETRY_URL=https://your-worker.example/v1/event

# Disable first-run prompt (CI)
export AUDIOBENCH_NO_PROMPT=1
```

Delete the consent file to reset:

```bash
rm ~/.config/audiobench/consent.json   # macOS / Linux
```

## Public aggregates

The dashboard at [Analytics](../analytics/index.html) pulls `GET /v1/summary` from the telemetry worker. That endpoint returns counts only (daily active install ids, top suites, top adapters, version distribution, error rate). No install ids are exposed.

## Retention

Events are deleted after **90 days** on the server. See [infra/telemetry/](https://github.com/THENIROCK/audiobench/tree/main/infra/telemetry) for the Worker + D1 schema and deploy steps.

## Default endpoint

```
https://audiobench-telemetry.thenirock.workers.dev/v1/event
```

Deploy your own worker before pointing users at a custom URL (see [Analytics setup](../guides/analytics-setup.md)).
