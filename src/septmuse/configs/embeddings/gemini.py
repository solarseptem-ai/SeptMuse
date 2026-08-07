#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Gemini 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class GeminiEmbedderConfig(BaseEmbedderConfig):
    """Google Gemini 嵌入配置。"""

    backend: str = Field(default="gemini")
    model: str = Field(default="models/gemini-embedding-001")
    embedding_dims: int = Field(default=768)
    output_dimensionality: int | None = Field(default=None)
