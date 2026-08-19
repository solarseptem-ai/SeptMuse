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
"""Ollama LLM provider。"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM, _robust_json_parse

logger = get_logger(__name__)


class OllamaLLM(LLM):
    """Ollama Chat provider。"""

    provider_name = "ollama"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen3.5:latest",
        host: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.model = model
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:7521")
        self._client = Client(host=self._host, timeout=30.0)
        logger.info("ollama_llm_ready", model=model, host=self._host)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Ollama Chat。"""
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

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """使用 Ollama format=json 返回结构化输出。"""
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
            )
            content = response["message"]["content"]
            return _robust_json_parse(content)
        except Exception:
            logger.debug("ollama_json_mode_failed", model=self.model)
        return super().complete_json(system_prompt, user_prompt)
