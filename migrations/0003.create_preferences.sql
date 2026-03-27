-- depends: 0002.create_facts

CREATE TABLE IF NOT EXISTS mneme.preferences (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    suggestion_id   TEXT,
    action          TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mneme_prefs_ws
    ON mneme.preferences(workspace_id, created_at DESC);
