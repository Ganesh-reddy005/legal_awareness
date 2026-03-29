from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str
    qdrant_api: str
    qdrant_collection: str = "legal_docs"

    # OpenRouter
    openrouter_key: str

    # Groq
    groq_api: str = ""
    groq_llm_model: str = "llama-3.3-70b-versatile"

    # LLM provider: "groq" or "openrouter"
    llm_provider: str = "groq"

    # Models (resolved from OpenRouter — used for embeddings only)
    embedding_model: str
    llm_model: str  # OpenRouter LLM model (used when llm_provider=openrouter)

    # Embedding dimensions for the chosen model
    # nvidia/llama-nemotron-embed-vl-1b-v2 outputs 2048-dim vectors
    embedding_dim: int = 2048

    # RAG config
    top_k: int = 5


settings = Settings()
