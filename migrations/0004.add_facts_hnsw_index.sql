-- depends: 0002.create_facts

CREATE INDEX IF NOT EXISTS idx_mneme_facts_embedding
    ON mneme.facts USING hnsw (embedding vector_cosine_ops);
