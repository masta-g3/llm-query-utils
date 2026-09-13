-- Additive: existing writers and readers continue to work during rollout.
ALTER TABLE papers.token_usage_logs
    ADD COLUMN IF NOT EXISTS project_id TEXT,
    ADD COLUMN IF NOT EXISTS reasoning_output_tokens INTEGER,
    ADD COLUMN IF NOT EXISTS total_cost NUMERIC;
