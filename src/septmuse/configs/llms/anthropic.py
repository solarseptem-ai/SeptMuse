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
"""Anthropic LLM 配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.llms.base import BaseLLMConfig


class AnthropicLLMConfig(BaseLLMConfig):
    """Anthropic Claude 配置。"""

    backend: str = Field(default="anthropic")
    model: str = Field(default="claude-3-5-haiku-latest")
    max_tokens: int = Field(default=4096)
