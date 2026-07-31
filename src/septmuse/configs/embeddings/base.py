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
"""嵌入配置基类 (借鉴 mem0 configs/embeddings/base.py BaseEmbedderConfig)。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaseEmbedderConfig(BaseModel):
    """嵌入通用配置。"""

    backend: str = Field(default="hash")
    model: str | None = Field(default=None, description="模型名")
    embedding_dims: int | None = Field(default=None, description="嵌入维度, None=自动检测")
    api_key: str | None = Field(default=None)
