# Mneme

**Mneme** is the memory service of [brainstack](https://github.com/andrerogers/brainstack). It manages conversational sessions, message history (with semantic embeddings), and stored facts — providing the long-term and working memory that Hive draws on during every chat request.

---

## Role in the pipeline

```
Hive (orchestration core)
  │  HTTP  POST /sessions/{id}/messages   ← append turn + embed
  │  HTTP  POST /recall                   ← semantic fact search
  │  HTTP  POST /sessions/recall          ← semantic message search
  ▼
Mneme (FastAPI) — sessions · messages · facts
  │
  ▼
PostgreSQL + pgvector (mneme.* schema)
  │  OpenRouter embeddings
  ▼
openai/text-embedding-3-small (1536 dims)
```

Hive calls Mneme directly over HTTP — Cortex does not proxy these calls.

---

## Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/sessions` | Create a new session |
| `GET` | `/sessions/{id}` | Fetch full message history for a session |
| `GET` | `/sessions` | List sessions, optionally filtered by `workspace_id` |
| `POST` | `/sessions/{id}/messages` | Append messages to a session (embeddings generated inline) |
| `DELETE` | `/sessions/{id}` | Hard-delete session and all messages (CASCADE); 204 / 404 |
| `POST` | `/sessions/recall` | Semantic search over message history via pgvector |
| `POST` | `/remember` | Embed and store a fact in `mneme.facts` |
| `POST` | `/recall` | Semantic search over stored facts via pgvector |
| `GET` | `/prefs/{workspace_id}` | Retrieve preference signals for a workspace |
| `POST` | `/prefs` | Upsert a preference signal |

---

## Directory layout

```
mneme/
├── mneme/
│   ├── app.py          FastAPI app, lifespan (pool open/close), route registration
│   ├── store.py        Async Postgres store — AsyncConnectionPool (psycopg-pool)
│   ├── models.py       Pydantic schemas for all routes
│   ├── embeddings.py   OpenRouter embedding client (text-embedding-3-small, batch ≤100)
│   └── config.py       Env-based config (DATABASE_URL, MNEME_PORT, OPENROUTER_API_KEY)
├── migrations/
│   ├── 0001.create_schema.sql              mneme schema + sessions + messages
│   ├── 0002.add_facts.sql                  mneme.facts table
│   ├── 0003.add_facts_embedding.sql        embedding vector(1536) on facts
│   ├── 0004.add_facts_hnsw_index.sql       HNSW index on facts.embedding
│   └── 0005.add_messages_embedding.sql     embedding vector(1536) + HNSW on messages
├── tests/
│   └── test_mneme_api.py   10 tests — health, sessions, messages, remember/recall
├── pyproject.toml
└── .env.example
```

---

## Setup

**Prerequisites:** Python 3.13, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector (`pgvector/pgvector:pg16`)

```bash
cd mneme
cp .env.example .env
# Edit .env — set DATABASE_URL and OPENROUTER_API_KEY
uv sync
uv run task dev      # starts uvicorn on port 8612 with --reload
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `MNEME_PORT` | `8612` | Port to bind |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter embeddings — `/remember` and `/sessions/recall` gracefully skip if unset |

---

## Schema

All tables live in the `mneme` schema in the shared PostgreSQL database.

| Table | Key columns |
|-------|-------------|
| `mneme.sessions` | `id UUID`, `workspace_id`, `created_at` |
| `mneme.messages` | `id`, `session_id`, `role`, `content`, `embedding vector(1536)`, `created_at` |
| `mneme.facts` | `id`, `content`, `embedding vector(1536)`, `tags`, `source`, `created_at` |
| `mneme.prefs` | `workspace_id`, `key`, `value`, `updated_at` |

HNSW indexes on `mneme.facts.embedding` and `mneme.messages.embedding` for sub-millisecond cosine similarity search.

---

## Connection pooling

Mneme uses `psycopg-pool AsyncConnectionPool(min_size=2, max_size=10)` opened lazily on first request and closed in the lifespan teardown. All queries go through `_run(fn)`:

```python
async with pool.connection() as conn:
    return await fn(conn)
```

---

## Embeddings

`POST /remember` and `POST /sessions/{id}/messages` both embed content via OpenRouter `openai/text-embedding-3-small` (1536 dims, batch ≤100). Embedding failures are caught and logged — the record is still written without a vector, so recall over those entries is skipped.

---

## Development

```bash
uv run task dev       # uvicorn --reload on port 8612
uv run task test      # pytest -v
uv run task lint      # ruff check .
uv run task typecheck # mypy mneme/
uv run task fmt       # ruff format .
uv run task check     # lint + typecheck + test
```

---

## Tech stack

| Layer | Library |
|-------|---------|
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | psycopg v3 async + psycopg-pool |
| Migrations | yoyo-migrations (isolated `_mneme_yoyo_*` tables) |
| Vector search | pgvector (1536-dim HNSW cosine) |
| Embeddings | OpenRouter `openai/text-embedding-3-small` |
| Observability | brainstack-optics (OTel) |

---

**Project devlog:** [andrerogers/vault — brainstack.md](https://github.com/andrerogers/vault/blob/master/projects/brainstack/brainstack.md)

---

MIT © 2025 [Andre Rogers](https://github.com/andrerogers)
