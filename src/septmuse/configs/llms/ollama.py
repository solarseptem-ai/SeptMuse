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
"""Ollama LLM 配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.llms.base import BaseLLMConfig


class OllamaLLMConfig(BaseLLMConfig):
    """Ollama 配置 (零配置 localhost:11434)。"""

    backend: str = Field(default="ollama")
    model: str = Field(default="qwen2.5:7b")
    host: str | None = Field(default=None, description="Ollama host, 默认 http://localhost:11434")
