#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Mock 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class MockEmbedderConfig(BaseEmbedderConfig):
    """Mock 嵌入配置 — 固定向量测试用。"""

    backend: str = Field(default="mock")
