-- depends: 0001.create_sessions_and_messages

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS mneme.facts (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT,
    content       TEXT NOT NULL,
    source        TEXT,
    tags          TEXT[] DEFAULT '{}',
    embedding     vector(1536),
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mneme_facts_ws
    ON mneme.facts(workspace_id);
