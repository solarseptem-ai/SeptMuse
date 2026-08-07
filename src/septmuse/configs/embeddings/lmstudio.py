#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""LM Studio 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class LMStudioEmbedderConfig(BaseEmbedderConfig):
    """LM Studio 嵌入配置。"""

    backend: str = Field(default="lmstudio")
    model: str = Field(default="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    embedding_dims: int = Field(default=1536)
