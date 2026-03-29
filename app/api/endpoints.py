"""
endpoints.py
------------
FastAPI routers for the Legal Awareness System.

Routes:
  POST /api/v1/ingest  — Extract, chunk, embed, and store the PDF
  POST /api/v1/query   — Answer a legal question via RAG
  GET  /api/v1/health  — Health check
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import IngestResponse, QueryRequest, QueryResponse
from app.services.embeddings import embed_texts
from app.services.pdf_extractor import extract_chunks
from app.services.qdrant_store import ensure_collection, upsert_chunks
from app.services.rag_chain import answer_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Path to the PDF — relative to project root where uvicorn is launched
_PDF_PATH = Path("data/dataset_main.pdf")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", tags=["System"])
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["Ingestion"],
    summary="Extract, embed, and store legal document chunks into Qdrant",
)
async def ingest() -> IngestResponse:
    """
    Processes `data/dataset_main.pdf`:
    1. Extracts and structurally chunks the document.
    2. Generates embeddings for all chunks via OpenRouter.
    3. Upserts vectors and metadata into Qdrant.

    Call this once (or whenever the PDF is updated).
    """
    if not _PDF_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found at {_PDF_PATH}",
        )

    # Step 1: Extract structural chunks
    logger.info("Extracting chunks from PDF…")
    chunks = extract_chunks(_PDF_PATH)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No chunks could be extracted from the PDF.",
        )
    logger.info(f"Extracted {len(chunks)} chunks.")

    # Step 2: Ensure Qdrant collection exists (recreate if dims changed)
    await ensure_collection(force_recreate=True)

    # Step 3: Embed in batches and upsert
    texts = [c.text for c in chunks]
    batch_size = 20  # Keep batches small for API rate limits
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1}/{-(-len(texts) // batch_size)}")
        vectors = await embed_texts(batch)
        all_vectors.extend(vectors)

    # Step 4: Upsert into Qdrant
    count = await upsert_chunks(chunks, all_vectors)
    logger.info(f"Upserted {count} points into Qdrant.")

    return IngestResponse(
        message="Ingestion complete.",
        chunks_ingested=count,
    )


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Query"],
    summary="Answer a legal question using RAG over stored legal documents",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieves semantically relevant legal context and answers the question
    strictly based on the retrieved text. No legal advice is provided.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    try:
        response = await answer_query(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Error in RAG pipeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {str(exc)}",
        ) from exc

    return response
