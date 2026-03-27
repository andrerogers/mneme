"""Configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
MNEME_PORT: int = int(os.environ.get("MNEME_PORT", "8612"))
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_EMBEDDINGS_URL: str = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
