from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None  # overrides settings.top_k if provided


class SourceChunk(BaseModel):
    part: Optional[str]
    article: Optional[str]
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int
