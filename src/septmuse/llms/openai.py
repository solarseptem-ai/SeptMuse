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
"""OpenAI LLM provider。

对齐 septmuse.llms.base.LLM ABC,
调用 OpenAI Chat Completions API。

用法:
    llm = OpenAILLM(api_key="sk-...", model="gpt-4o")
    response = llm.complete(system_prompt, user_prompt)

零配置: 从环境变量 OPENAI_API_KEY 读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class OpenAILLM(LLM):
    """OpenAI Chat Completions provider。

    零配置: 从 OPENAI_API_KEY 环境变量读取。
    自定义: OpenAILLM(api_key="sk-...", model="gpt-4o")。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-required"
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        client_kwargs.update(kwargs)

        self._client = OpenAI(**client_kwargs)
        logger.info("openai_llm_ready", model=model, base_url=resolved_base_url or "default")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI Chat Completions (对齐 LLM ABC)。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.debug("openai_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("openai_complete_failed", error=str(e))
            raise
