#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""Azure OpenAI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class AzureOpenAIEmbedderConfig(BaseEmbedderConfig):
    """Azure OpenAI 嵌入配置。"""

    backend: str = Field(default="azure_openai")
    model: str = Field(default="text-embedding-3-small")
    azure_deployment: str | None = Field(default=None)
    azure_endpoint: str | None = Field(default=None)
    api_version: str | None = Field(default=None)
    embedding_dims: int = Field(default=1536)
