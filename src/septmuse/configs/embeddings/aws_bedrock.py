#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license)
"""AWS Bedrock 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class AWSBedrockEmbedderConfig(BaseEmbedderConfig):
    """AWS Bedrock 嵌入配置。"""

    backend: str = Field(default="aws_bedrock")
    model: str = Field(default="amazon.titan-embed-text-v1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    aws_session_token: str | None = Field(default=None)
    aws_region: str = Field(default="us-west-2")
    embedding_dims: int | None = Field(default=None)
