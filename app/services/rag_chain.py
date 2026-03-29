"""
rag_chain.py
------------
RAG pipeline:
  1. Embed the user's question (via OpenRouter embeddings)
  2. Retrieve top-k relevant chunks from Qdrant
  3. Build a structured system prompt with the retrieved context
  4. Call the LLM via the unified llm_client (Groq or OpenRouter)
  5. Return the answer and the source chunks used
"""

from app.models.schemas import QueryResponse, SourceChunk
from app.services.embeddings import embed_single
from app.services.llm_client import chat_complete
from app.services.qdrant_store import search_similar


_SYSTEM_PROMPT = """You are a Situational Legal Awareness assistant for Indian law.

STRICT RULES:
1. Answer ONLY using the legal context provided and summarise it so people can easily understand.
2. Do NOT provide legal advice or personal opinions.
3. Do NOT use any knowledge outside of the provided context.
4. If the answer is not present in the context, say:
   "I could not find relevant information in the provided legal documents." and then give a general awareness response based on your own knowledge.
5. Always cite the Part and Article from the context in your response.
6. Be precise, clear, and concise.

CONTEXT:
{context}
"""


async def answer_query(
    question: str,
    top_k: int | None = None,
) -> QueryResponse:
    """
    Full RAG pipeline: question → context retrieval → LLM answer.

    Args:
        question: User's natural language legal question.
        top_k:    How many context chunks to retrieve (falls back to config).

    Returns:
        QueryResponse with answer text and source chunks.
    """
    # 1. Embed the question
    query_vector = await embed_single(question)

    # 2. Retrieve similar chunks from Qdrant
    results = await search_similar(query_vector, top_k=top_k)

    # 3. Build context block from retrieved chunks
    context_parts: list[str] = []
    source_chunks: list[SourceChunk] = []

    for hit in results:
        payload = hit.payload or {}
        part = payload.get("part", "Unknown")
        article = payload.get("article", "Unknown")
        text = payload.get("text", "")
        score = round(float(hit.score), 4)

        context_parts.append(f"[{part} | {article}]\n{text}")
        source_chunks.append(
            SourceChunk(part=part, article=article, text=text, score=score)
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # 4. Call LLM via unified provider (Groq or OpenRouter)
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT.format(context=context_block),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    answer = await chat_complete(messages, temperature=0.2, max_tokens=1024)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=source_chunks,
    )
