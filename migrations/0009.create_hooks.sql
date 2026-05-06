-- depends: 0008.create_jobs

CREATE TABLE IF NOT EXISTS mneme.hooks (
    name        TEXT PRIMARY KEY,
    on_subject  TEXT NOT NULL,
    run         TEXT NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);
