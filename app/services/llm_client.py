"""
llm_client.py
-------------
Unified LLM client that routes to either Groq or OpenRouter
based on the LLM_PROVIDER setting in .env.

To switch provider:
    LLM_PROVIDER=groq        → uses GROQ_API + GROQ_LLM_MODEL  (fast)
    LLM_PROVIDER=openrouter  → uses OPENROUTER_KEY + LLM_MODEL  (flexible)
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "get_key": lambda: settings.groq_api,
        "get_model": lambda: settings.groq_llm_model,
        "extra_headers": {},
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "get_key": lambda: settings.openrouter_key,
        "get_model": lambda: settings.llm_model,
        "extra_headers": {
            "HTTP-Referer": "https://legal-awareness-system.local",
            "X-Title": "Situational Legal Awareness System",
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def chat_complete(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Send a chat completion request to the configured LLM provider.

    Args:
        messages:    OpenAI-format message list (system + user).
        temperature: Sampling temperature.
        max_tokens:  Max tokens in the response.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        ValueError: If the configured provider is unknown.
        httpx.HTTPStatusError: On HTTP errors from the provider.
    """
    provider_name = settings.llm_provider.lower()

    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider_name}'. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )

    cfg = _PROVIDERS[provider_name]
    api_key = cfg["get_key"]()
    model = cfg["get_model"]()

    logger.info(f"LLM request → provider={provider_name}, model={model}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **cfg["extra_headers"],
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(cfg["url"], headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    answer = data["choices"][0]["message"]["content"].strip()
    logger.info(f"LLM response received ({len(answer)} chars)")
    return answer
