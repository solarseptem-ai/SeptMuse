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
"""Cohere 云重排器配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.rerankers.base import BaseRerankerConfig


class CohereRerankerConfig(BaseRerankerConfig):
    """Cohere reranker 配置。

    需要 COHERE_API_KEY 环境变量或构造参数传入 api_key。
    模型默认 rerank-v3.5。
    """

    backend: str = Field(default="cohere")
    api_key: str = Field(default="", description="Cohere API key (空则读 COHERE_API_KEY 环境变量)")
    model: str = Field(default="rerank-v3.5", description="Cohere rerank 模型名")
