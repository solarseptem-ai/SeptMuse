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
"""Groq LLM provider — 超低延迟推理。

用法:
    llm = GroqLLM(api_key="...", model="llama-3.1-70b-versatile")
    result = llm.complete(system_prompt, user_prompt)

零配置: 从 GROQ_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class GroqLLM(LLM):
    """Groq LLM provider。零配置: 从 GROQ_API_KEY 环境变量读取 key。"""

    provider_name = "groq"

    def __init__(self, api_key: str | None = None, model: str = "llama-3.1-70b-versatile", **kwargs: Any) -> None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("groq package required: pip install septmuse[groq]") from e

        self.model = model
        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"), **kwargs)
        logger.info("groq_llm_ready", model=model)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Groq Chat Completions (对齐 LLM ABC)。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.debug("groq_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("groq_complete_failed", error=str(e))
            raise
