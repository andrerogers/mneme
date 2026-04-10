-- depends: 0003.create_preferences

ALTER TABLE mneme.preferences
    ADD COLUMN IF NOT EXISTS task_summary TEXT,
    ADD COLUMN IF NOT EXISTS feedback_text TEXT;
