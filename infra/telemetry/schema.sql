CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  install_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  event TEXT NOT NULL,
  version TEXT NOT NULL,
  py_minor TEXT,
  os TEXT,
  country TEXT,
  suite TEXT,
  adapter TEXT,
  duration_ms INTEGER,
  ok INTEGER
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_install ON events(install_id, ts);
