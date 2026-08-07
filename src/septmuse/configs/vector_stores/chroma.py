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
"""Chroma 向量存储配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from septmuse.configs.vector_stores.base import BaseVectorStoreConfig


def _default_persist_path() -> str:
    return str(Path.home() / ".septmuse" / "chroma")


class ChromaVectorConfig(BaseVectorStoreConfig):
    """Chroma 向量存储配置 (本地持久化)。"""

    persist_path: str = Field(description="Chroma 持久化路径", default_factory=_default_persist_path)
    collection_name: str = Field(default="septmuse")
    embedding_model_dims: int = Field(default=512)
