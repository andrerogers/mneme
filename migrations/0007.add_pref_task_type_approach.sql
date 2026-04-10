-- depends: 0006.add_pref_task_summary_feedback

ALTER TABLE mneme.preferences
    ADD COLUMN IF NOT EXISTS task_type TEXT,
    ADD COLUMN IF NOT EXISTS approach_notes TEXT;
