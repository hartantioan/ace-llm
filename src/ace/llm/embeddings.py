import os
from enum import StrEnum
from typing import Any

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# Add more imports as you add providers (e.g., CohereEmbeddings, HuggingFaceEmbeddings, etc.)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class EmbeddingsEnum(StrEnum):
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    # Local embedding models served by Ollama (`ollama pull nomic-embed-text`)
    OLLAMA_NOMIC_EMBED_TEXT = "nomic-embed-text"
    # Add more as needed, e.g. COHERE = "cohere-embed-v3", HUGGINGFACE = "sentence-transformers/all-MiniLM-L6-v2"


EMBEDDING_PROVIDERS = {
    EmbeddingsEnum.OPENAI_3_SMALL: "openai",
    EmbeddingsEnum.OPENAI_3_LARGE: "openai",
    EmbeddingsEnum.OLLAMA_NOMIC_EMBED_TEXT: "ollama",
    # Add more as needed
}

EMBEDDING_NAMES = {
    EmbeddingsEnum.OPENAI_3_SMALL: "OpenAI Embeddings 3 Small",
    EmbeddingsEnum.OPENAI_3_LARGE: "OpenAI Embeddings 3 Large",
    EmbeddingsEnum.OLLAMA_NOMIC_EMBED_TEXT: "Nomic Embed Text (Ollama, local)",
    # Add more as needed
}


def create_embedding(model: EmbeddingsEnum, **kwargs: Any):
    """
    Factory: Instantiate and return a LangChain embedding model for the given enum, with backend-specific config.
    Args:
        model: EmbeddingsEnum value (e.g. EmbeddingsEnum.OPENAI_3_SMALL)
        **kwargs: Passed to the backend constructor (e.g. api_key, model, etc)
    Returns:
        LangChain Embeddings instance (e.g. OpenAIEmbeddings, CohereEmbeddings, etc)
    Raises:
        ValueError if the provider is unknown or required kwargs are missing.
    """
    # TODO: need to debug what's going on here with similarity score error
    provider = EMBEDDING_PROVIDERS[model]
    if provider == "openai":
        return OpenAIEmbeddings(model=model.value, **kwargs)
    elif provider == "ollama":
        kwargs.setdefault("base_url", os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        return OllamaEmbeddings(model=model.value, **kwargs)
    # Add more providers here
    else:
        raise ValueError(f"Unknown or unsupported embedding provider: {provider}")
