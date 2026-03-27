"""Embedding client — calls OpenRouter for text-embedding-3-small."""

from __future__ import annotations

import logging

import httpx

from mneme.config import (
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_EMBEDDINGS_URL,
)

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter (OpenAI-compatible endpoint).

    Returns a list of float vectors, one per input text.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot generate embeddings")

    client = _get_client()
    resp = await client.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    # OpenAI embeddings response: { data: [ { embedding: [...], index: 0 }, ... ] }
    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]
