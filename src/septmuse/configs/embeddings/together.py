#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Together AI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class TogetherEmbedderConfig(BaseEmbedderConfig):
    """Together AI 嵌入配置。"""

    backend: str = Field(default="together")
    model: str = Field(default="intfloat/multilingual-e5-large-instruct")
    embedding_dims: int = Field(default=1024)
