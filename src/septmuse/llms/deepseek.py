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
"""DeepSeek LLM provider — OpenAI 兼容 API。

用法:
    llm = DeepSeekLLM(api_key="...", model="deepseek-chat")
    result = llm.complete(system_prompt, user_prompt)

零配置: 从 DEEPSEEK_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class DeepSeekLLM(LLM):
    """DeepSeek LLM provider（OpenAI 兼容 API）。零配置: 从 DEEPSEEK_API_KEY 环境变量读取 key。"""

    provider_name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
            **kwargs,
        )
        logger.info("deepseek_llm_ready", model=model)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DeepSeek Chat Completions (对齐 LLM ABC)。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.debug("deepseek_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("deepseek_complete_failed", error=str(e))
            raise
