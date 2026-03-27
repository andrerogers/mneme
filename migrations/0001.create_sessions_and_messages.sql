-- depends:

CREATE SCHEMA IF NOT EXISTS mneme;

CREATE TABLE IF NOT EXISTS mneme.sessions (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    archived      BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_mneme_sessions_ws
    ON mneme.sessions(workspace_id, archived, created_at DESC);

CREATE TABLE IF NOT EXISTS mneme.messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT REFERENCES mneme.sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mneme_messages_session
    ON mneme.messages(session_id, created_at);
