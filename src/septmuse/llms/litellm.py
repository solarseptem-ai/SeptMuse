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
"""litellm 统一 LLM 代理 — 一个依赖覆盖 100+ provider。

用法:
    llm = LitellmLLM(model="groq/llama-3.1-70b-versatile", api_key="...")
    result = llm.complete(system_prompt, user_prompt)

model 格式: "provider/model"，如 "groq/llama-3.1-70b-versatile"。
零配置: 从环境变量读取 api_key（按 provider 前缀）。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class LitellmLLM(LLM):
    """litellm 统一 LLM provider。

    model 格式: "provider/model"，如 "groq/llama-3.1-70b-versatile"。
    零配置: 从环境变量读取 api_key（按 provider 前缀）。
    """

    provider_name = "litellm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm package required: pip install septmuse[litellm]") from e

        self._litellm = litellm
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._extra_kwargs = kwargs
        logger.info("litellm_llm_ready", model=model)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """委托 litellm.completion。"""
        try:
            response = self._litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._api_key,
                api_base=self._base_url,
                **self._extra_kwargs,
            )
            content = response.choices[0].message.content or ""
            logger.debug("litellm_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("litellm_complete_failed", error=str(e))
            raise
