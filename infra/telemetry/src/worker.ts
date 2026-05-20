/**
 * audiobench opt-in telemetry worker.
 * POST /v1/event  — ingest anonymous CLI events
 * GET  /v1/summary — public aggregates (no PII)
 */

export interface Env {
  DB: D1Database;
  RETENTION_DAYS?: string;
}

interface EventBody {
  install_id?: string;
  event?: string;
  version?: string;
  py_minor?: string;
  os?: string;
  ts?: number;
  suite?: string;
  adapter?: string;
  duration_ms?: number;
  ok?: number;
}

const MAX_FIELD = 128;

function trim(s: string | undefined, max = MAX_FIELD): string | undefined {
  if (s == null) return undefined;
  const t = String(s).slice(0, max);
  return t || undefined;
}

function corsHeaders(): HeadersInit {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

async function pruneOld(db: D1Database, retentionDays: number): Promise<void> {
  const cutoff = Math.floor(Date.now() / 1000) - retentionDays * 86400;
  await db.prepare("DELETE FROM events WHERE ts < ?").bind(cutoff).run();
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    const url = new URL(request.url);
    const retention = parseInt(env.RETENTION_DAYS || "90", 10);

    if (request.method === "POST" && url.pathname === "/v1/event") {
      let body: EventBody;
      try {
        body = (await request.json()) as EventBody;
      } catch {
        return new Response("invalid json", { status: 400, headers: corsHeaders() });
      }

      const installId = trim(body.install_id, 64);
      const event = trim(body.event, 64);
      const version = trim(body.version, 32);
      if (!installId || !event || !version) {
        return new Response("missing required fields", { status: 400, headers: corsHeaders() });
      }

      const country = request.headers.get("cf-ipcountry") || undefined;
      const ts = typeof body.ts === "number" ? body.ts : Math.floor(Date.now() / 1000);

      await env.DB.prepare(
        `INSERT INTO events (
          install_id, ts, event, version, py_minor, os, country,
          suite, adapter, duration_ms, ok
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
        .bind(
          installId,
          ts,
          event,
          version,
          trim(body.py_minor, 16) ?? null,
          trim(body.os, 32) ?? null,
          country ?? null,
          trim(body.suite, 64) ?? null,
          trim(body.adapter, 64) ?? null,
          typeof body.duration_ms === "number" ? body.duration_ms : null,
          typeof body.ok === "number" ? body.ok : null
        )
        .run();

      // Prune occasionally (1% of requests)
      if (Math.random() < 0.01) {
        await pruneOld(env.DB, retention);
      }

      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (request.method === "GET" && url.pathname === "/v1/summary") {
      const since = Math.floor(Date.now() / 1000) - 30 * 86400;

      const daily = await env.DB.prepare(
        `SELECT date(ts, 'unixepoch') AS day, COUNT(DISTINCT install_id) AS dau, COUNT(*) AS events
         FROM events WHERE ts >= ? GROUP BY day ORDER BY day`
      )
        .bind(since)
        .all();

      const suites = await env.DB.prepare(
        `SELECT suite AS name, COUNT(*) AS count FROM events
         WHERE ts >= ? AND suite IS NOT NULL GROUP BY suite ORDER BY count DESC LIMIT 10`
      )
        .bind(since)
        .all();

      const adapters = await env.DB.prepare(
        `SELECT adapter AS name, COUNT(*) AS count FROM events
         WHERE ts >= ? AND adapter IS NOT NULL GROUP BY adapter ORDER BY count DESC LIMIT 10`
      )
        .bind(since)
        .all();

      const versions = await env.DB.prepare(
        `SELECT version AS name, COUNT(*) AS count FROM events
         WHERE ts >= ? GROUP BY version ORDER BY count DESC LIMIT 10`
      )
        .bind(since)
        .all();

      const countries = await env.DB.prepare(
        `SELECT country AS name, COUNT(*) AS count FROM events
         WHERE ts >= ? AND country IS NOT NULL GROUP BY country ORDER BY count DESC LIMIT 15`
      )
        .bind(since)
        .all();

      const errors = await env.DB.prepare(
        `SELECT COUNT(*) AS total,
                SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failed
         FROM events WHERE ts >= ? AND event LIKE 'cmd.%'`
      )
        .bind(since)
        .first<{ total: number; failed: number }>();

      const payload = {
        generated_at: new Date().toISOString(),
        window_days: 30,
        daily_active: daily.results ?? [],
        top_suites: suites.results ?? [],
        top_adapters: adapters.results ?? [],
        top_versions: versions.results ?? [],
        by_country: countries.results ?? [],
        cmd_stats: errors ?? { total: 0, failed: 0 },
      };

      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: {
          ...corsHeaders(),
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=300",
        },
      });
    }

    return new Response("not found", { status: 404, headers: corsHeaders() });
  },
};
