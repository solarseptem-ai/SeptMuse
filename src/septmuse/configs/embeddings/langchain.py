#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Langchain 嵌入配置。"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class LangchainEmbedderConfig(BaseEmbedderConfig):
    """Langchain 嵌入配置 — model 字段是 Embeddings 实例 (非字符串)。"""

    backend: str = Field(default="langchain")
    model: Any = Field(default=None, description="langchain Embeddings 实例")
    embedding_dims: int = Field(default=768)
