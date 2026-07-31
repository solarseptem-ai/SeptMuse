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
"""Ollama LLM provider (借鉴 mem0 llms/ollama.py 模式)。

对齐 septmuse.llms.base.LLM ABC,
调用 Ollama Chat API。

用法:
    llm = OllamaLLM(model="qwen2.5:7b")
    response = llm.complete(system_prompt, user_prompt)

零配置: 默认 localhost:11434, 无需 API key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class OllamaLLM(LLM):
    """Ollama Chat provider (借鉴 mem0 llms/ollama.py)。

    零配置: 默认 http://localhost:11434, 无需 API key。
    自定义: OllamaLLM(model="qwen2.5:7b", host="http://gpu-server:11434")。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen2.5:7b",
        host: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.model = model
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=self._host)
        logger.info("ollama_llm_ready", model=model, host=self._host)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Ollama Chat (对齐 LLM ABC)。"""
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response["message"]["content"]
            logger.debug("ollama_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("ollama_complete_failed", error=str(e))
            raise
