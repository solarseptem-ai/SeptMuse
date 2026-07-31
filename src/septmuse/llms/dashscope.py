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
"""DashScope (Qwen) LLM provider (SeptMuse 创新, 对齐中国用户)。

对齐 septmuse.llms.base.LLM ABC,
调用 DashScope Generation API。

用法:
    llm = DashScopeLLM(api_key="sk-ds-...", model="qwen-plus")
    response = llm.complete(system_prompt, user_prompt)

零配置: 从 DASHSCOPE_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class DashScopeLLM(LLM):
    """DashScope (Qwen) provider (SeptMuse 创新, 对齐中国用户)。

    零配置: 从 DASHSCOPE_API_KEY 环境变量读取。
    自定义: DashScopeLLM(api_key="sk-ds-...", model="qwen-plus")。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-plus",
        **kwargs: Any,
    ) -> None:
        try:
            import dashscope
        except ImportError as e:
            raise ImportError("dashscope package required: pip install septmuse[dashscope]") from e

        self.model = model
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not set: pass api_key or set env var")

        self._dashscope = dashscope
        logger.info("dashscope_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DashScope Generation API (对齐 LLM ABC)。"""
        try:
            response = self._dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._api_key,
                result_format="message",
            )
            content = response.output.choices[0].message.content
            logger.debug("dashscope_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("dashscope_complete_failed", error=str(e))
            raise
