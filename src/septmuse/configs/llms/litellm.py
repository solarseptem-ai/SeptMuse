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
"""litellm LLM 配置 — 一个依赖覆盖 100+ provider。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.llms.base import BaseLLMConfig


class LitellmLLMConfig(BaseLLMConfig):
    """litellm 统一 LLM 配置 — 一个依赖覆盖 100+ provider。"""

    backend: str = Field(default="litellm")
    model: str = Field(default="gpt-4o-mini")
    base_url: str | None = Field(default=None, description="自定义 API 端点")
