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
"""Anthropic LLM provider。"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class AnthropicLLM(LLM):
    """Anthropic Messages provider。"""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install septmuse[anthropic]") from e

        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set: pass api_key or set env var")

        self._client = Anthropic(api_key=self._api_key, timeout=30.0)
        logger.info("anthropic_llm_ready", model=model)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Anthropic Messages API。"""
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self._max_tokens,
            )
            content = response.content[0].text if response.content else ""
            logger.debug("anthropic_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("anthropic_complete_failed", error=str(e))
            raise
