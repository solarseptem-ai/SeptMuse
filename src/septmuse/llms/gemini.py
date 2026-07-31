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
"""Google Gemini LLM provider。

用法:
    llm = GeminiLLM(api_key="...", model="gemini-1.5-flash")
    result = llm.complete(system_prompt, user_prompt)

零配置: 从 GEMINI_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class GeminiLLM(LLM):
    """Google Gemini LLM provider。零配置: 从 GEMINI_API_KEY 环境变量读取 key。"""

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash", **kwargs: Any) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("google-generativeai required: pip install septmuse[gemini]") from e

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._genai = genai
        self.model = model
        self._model = genai.GenerativeModel(model)
        logger.info("gemini_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Gemini generate_content (对齐 LLM ABC)。"""
        try:
            response = self._model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            content = response.text or ""
            logger.debug("gemini_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("gemini_complete_failed", error=str(e))
            raise
