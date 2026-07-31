#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Qdrant 向量存储配置 (借鉴 mem0 QdrantConfig)。"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, model_validator

from septmuse.configs.vector_stores.base import BaseVectorStoreConfig


class QdrantVectorConfig(BaseVectorStoreConfig):
    """Qdrant 向量存储配置。"""

    collection_name: str = Field(default="septmuse")
    embedding_model_dims: int = Field(default=384)
    host: str | None = Field(default=None)
    port: int | None = Field(default=None)
    url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    path: str | None = Field(default=None)
    https: bool | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_host_port_or_path(cls, values: dict[str, Any]) -> dict[str, Any]:
        host = values.get("host")
        port = values.get("port")
        path = values.get("path")
        url = values.get("url")
        api_key = values.get("api_key")
        if not path and not (host and port) and not (url and api_key):
            raise ValueError("Either 'host' and 'port' or 'url' and 'api_key' or 'path' must be provided.")
        return values
