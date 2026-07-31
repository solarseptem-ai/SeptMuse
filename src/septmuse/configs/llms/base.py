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
"""LLM 配置基类 (借鉴 mem0 configs/llms/base.py BaseLlmConfig)。

通用参数: model / temperature / api_key / max_tokens。
provider 特有参数在各子类中添加。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaseLLMConfig(BaseModel):
    """LLM 通用配置。"""

    backend: str = Field(default="", description="provider 后端标识, 如 openai/ollama/anthropic/dashscope")
    model: str | None = Field(default=None, description="模型名")
    temperature: float = Field(default=0.1, description="采样温度")
    api_key: str | None = Field(default=None, description="API key")
    max_tokens: int = Field(default=2000, description="最大 token 数")
    top_p: float = Field(default=0.1, description="核采样参数")
