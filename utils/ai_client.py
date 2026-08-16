"""
LangChain-based AI client factory.

Provides a unified interface for Claude (Anthropic) and Ollama backends.
Use get_llm() for chat completions and get_embeddings() for vector embeddings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from config.settings import (
    AI_BACKEND,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MODEL_FAST,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EMBED_MODEL,
    EMBED_BACKEND,
    HUGGINGFACE_EMBED_MODEL,
)


def get_llm(temperature: float = 0.3, fast: bool = False) -> BaseChatModel:
    """
    Return a LangChain chat model for the configured backend.

    Args:
        temperature: Sampling temperature (0 = deterministic).
        fast: Use the faster/cheaper model variant (Claude haiku vs sonnet).
    """
    if AI_BACKEND == "claude":
        from langchain_anthropic import ChatAnthropic
        model = CLAUDE_MODEL_FAST if fast else CLAUDE_MODEL
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=4096,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=8192,
            num_predict=4096,
        )


def get_embeddings() -> Embeddings:
    """
    Return the appropriate embeddings model based on backend configuration.

    - Ollama backend: uses OllamaEmbeddings (nomic-embed-text)
    - Claude backend + EMBED_BACKEND=ollama: uses OllamaEmbeddings
    - Claude backend + EMBED_BACKEND=huggingface: uses HuggingFaceEmbeddings (local, ~80MB)
    """
    use_ollama_embed = (AI_BACKEND == "ollama") or (EMBED_BACKEND == "ollama")

    if use_ollama_embed:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def check_llm_availability() -> tuple[bool, str]:
    """
    Test LLM connectivity without spending tokens.
    Returns (is_available, error_message).
    """
    try:
        llm = get_llm(temperature=0, fast=True)
        # Minimal invoke to verify credentials and connectivity
        llm.invoke([HumanMessage(content="Hi")])
        return True, ""
    except Exception as exc:
        return False, str(exc)


def get_backend_display_name() -> str:
    """Human-readable description of the active backend."""
    if AI_BACKEND == "claude":
        return f"Claude ({CLAUDE_MODEL})"
    return f"Ollama ({OLLAMA_MODEL} @ {OLLAMA_BASE_URL})"


def get_embed_display_name() -> str:
    """Human-readable description of the active embeddings backend."""
    use_ollama_embed = (AI_BACKEND == "ollama") or (EMBED_BACKEND == "ollama")
    if use_ollama_embed:
        return f"Ollama ({OLLAMA_EMBED_MODEL})"
    return f"HuggingFace ({HUGGINGFACE_EMBED_MODEL})"
