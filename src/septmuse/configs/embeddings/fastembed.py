#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""FastEmbed 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class FastEmbedEmbedderConfig(BaseEmbedderConfig):
    """FastEmbed 嵌入配置。"""

    backend: str = Field(default="fastembed")
    model: str = Field(default="thenlper/gte-large")
    embedding_dims: int | None = Field(default=None)
