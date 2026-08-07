#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Ollama 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class OllamaEmbedderConfig(BaseEmbedderConfig):
    """Ollama 嵌入配置。"""

    backend: str = Field(default="ollama")
    model: str = Field(default="nomic-embed-text")
    ollama_base_url: str = Field(default="http://localhost:11434")
    embedding_dims: int = Field(default=512)
