"""
qdrant_store.py
---------------
Manages the Qdrant vector database:
  - Initialise collection (idempotent)
  - Upsert document chunks
  - Search for semantically similar chunks
"""

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.core.config import settings
from app.services.pdf_extractor import LegalChunk


# ---------------------------------------------------------------------------
# Client (module-level singleton)
# ---------------------------------------------------------------------------

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api,
            check_compatibility=False,
        )
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

async def ensure_collection(force_recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Pass force_recreate=True to drop and recreate (e.g. after changing dim).
    """
    client = get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if settings.qdrant_collection in names:
        if force_recreate:
            await client.delete_collection(settings.qdrant_collection)
        else:
            return  # already exists, nothing to do

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

async def upsert_chunks(
    chunks: list[LegalChunk],
    vectors: list[list[float]],
) -> int:
    """
    Upsert LegalChunks with their pre-computed embedding vectors.

    Returns:
        Number of points upserted.
    """
    client = get_client()

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        payload: dict[str, Any] = {
            "part": chunk.part,
            "article": chunk.article,
            "text": chunk.text,
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    # Upload in batches of 100 to avoid request size limits
    batch_size = 100
    for i in range(0, len(points), batch_size):
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch_size],
        )

    return len(points)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def search_similar(
    query_vector: list[float],
    top_k: int | None = None,
) -> list[ScoredPoint]:
    """
    Retrieve the top-k most semantically similar chunks.
    Uses query_points() — the current API in qdrant-client >= 1.7.

    Args:
        query_vector: Embedding of the user's question.
        top_k:        Number of results to return (defaults to settings.top_k).

    Returns:
        List of ScoredPoint objects with payload and score.
    """
    client = get_client()
    k = top_k or settings.top_k

    response = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    return response.points
