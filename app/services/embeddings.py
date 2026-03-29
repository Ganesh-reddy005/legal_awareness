"""
embeddings.py
-------------
Generates text embeddings via the OpenRouter API using the configured model.

OpenRouter compatible with OpenAI SDK format for embeddings:
  POST https://openrouter.ai/api/v1/embeddings
"""

import httpx
from app.core.config import settings


_OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
_HEADERS = {
    "Authorization": f"Bearer {settings.openrouter_key}",
    "Content-Type": "application/json",
}


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts and return a list of float vectors.

    Args:
        texts: List of strings to embed. Keep batch size reasonable
               (OpenRouter may have per-request limits).

    Returns:
        List of embedding vectors in the same order as input texts.
    """
    if not texts:
        return []

    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _OPENROUTER_EMBED_URL,
            headers=_HEADERS,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    # OpenAI-compatible response format:
    # { "data": [{"embedding": [...], "index": 0}, ...], "model": "...", "usage": {...} }
    embeddings = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in embeddings]


async def embed_single(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    vectors = await embed_texts([text])
    return vectors[0]
