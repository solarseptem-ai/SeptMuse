#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""HuggingFace 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class HuggingFaceEmbedderConfig(BaseEmbedderConfig):
    """HuggingFace 嵌入配置 (本地 ST 或 TEI server)。"""

    backend: str = Field(default="huggingface")
    model: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    huggingface_base_url: str | None = Field(default=None, description="TEI server URL, None=本地 ST")
    model_kwargs: dict = Field(default_factory=dict)
    embedding_dims: int | None = Field(default=None)
