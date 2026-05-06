-- depends: 0007.add_pref_task_type_approach

CREATE TABLE IF NOT EXISTS mneme.jobs (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    payload          JSONB NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    idempotency_key  TEXT UNIQUE,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now(),
    result           JSONB
);

CREATE INDEX IF NOT EXISTS idx_mneme_jobs_status ON mneme.jobs(status);
CREATE INDEX IF NOT EXISTS idx_mneme_jobs_kind   ON mneme.jobs(kind);
