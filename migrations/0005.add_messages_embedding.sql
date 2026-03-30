-- depends: 0004.add_facts_hnsw_index

ALTER TABLE mneme.messages ADD COLUMN IF NOT EXISTS embedding vector(1536);

CREATE INDEX IF NOT EXISTS idx_mneme_messages_embedding
    ON mneme.messages USING hnsw (embedding vector_cosine_ops);
