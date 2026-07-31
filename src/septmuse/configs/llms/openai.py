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
"""OpenAI LLM 配置 (借鉴 mem0 OpenAIConfig)。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.llms.base import BaseLLMConfig


class OpenAILLMConfig(BaseLLMConfig):
    """OpenAI 配置 (支持 base_url 兼容端点)。"""

    backend: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    base_url: str | None = Field(default=None, description="OpenAI 兼容端点, 如 http://localhost:7521/v1")
