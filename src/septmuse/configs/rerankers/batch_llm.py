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
"""LLM 批量打分重排器配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.rerankers.base import BaseRerankerConfig


class BatchLLMRerankerConfig(BaseRerankerConfig):
    """LLM 批量打分重排器配置。

    一次请求对多个文档打分, 减少 API 调用。
    解析失败时回退到逐条打分。
    """

    backend: str = Field(default="batch_llm")
    max_input_len: int = Field(default=2000, description="单条输入截断长度")
    max_batch_size: int = Field(default=10, description="单次批量打分文档数上限")
