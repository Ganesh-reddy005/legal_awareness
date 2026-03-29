"""
main.py
-------
FastAPI application entry point.
Run with: uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Situational Legal Awareness System (India)",
    description=(
        "A RAG-based system that answers legal awareness questions "
        "strictly grounded in Indian legal documents. "
        "This service provides legal information, NOT legal advice."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow local frontend / testing tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "service": "Situational Legal Awareness System (India)",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "query": "POST /api/v1/query",
            "health": "GET  /api/v1/health",
        },
    }
