#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Vertex AI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class VertexAIEmbedderConfig(BaseEmbedderConfig):
    """Google Vertex AI 嵌入配置。"""

    backend: str = Field(default="vertexai")
    model: str = Field(default="gemini-embedding-001")
    embedding_dims: int = Field(default=256)
    vertex_credentials_json: str | None = Field(default=None)
    memory_add_embedding_type: str = Field(default="RETRIEVAL_DOCUMENT")
    memory_search_embedding_type: str = Field(default="RETRIEVAL_QUERY")
    memory_update_embedding_type: str = Field(default="RETRIEVAL_DOCUMENT")
