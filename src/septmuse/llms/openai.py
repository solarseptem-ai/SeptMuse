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
"""OpenAI LLM provider。"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM, _robust_json_parse

logger = get_logger(__name__)


class OpenAILLM(LLM):
    """OpenAI Chat Completions provider。"""

    provider_name = "openai"

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
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("SEPTMUSE_BASE_URL") or os.getenv("SEPTMUSE_LLM_BASE_URL")

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        # 只传递合法的 OpenAI client 构造参数, 过滤掉 LLM 配置字段 (temperature/max_tokens/top_p)
        for k in ("timeout", "max_retries", "default_headers", "default_query", "http_client"):
            if k in kwargs:
                client_kwargs[k] = kwargs[k]
        client_kwargs.setdefault("timeout", 30.0)
        self._client = OpenAI(**client_kwargs)
        logger.info("openai_llm_ready", model=model, base_url=resolved_base_url or "default")

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI Chat Completions。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 先尝试非流式
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            content = response.choices[0].message.content
            if content is not None:
                logger.debug("openai_complete_done", model=self.model, response_len=len(content))
                return content
        except Exception as e:
            logger.debug("openai_nonstream_failed", error=str(e)[:200])

        # 降级到流式
        try:
            logger.debug("openai_stream_fallback", model=self.model)
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            full = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full += chunk.choices[0].delta.content
            logger.debug("openai_complete_done", model=self.model, response_len=len(full))
            return full
        except Exception as e:
            logger.error("openai_complete_failed", error=str(e))
            raise

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """使用 OpenAI JSON mode 返回结构化输出。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            content = response.choices[0].message.content
            if content is not None:
                return _robust_json_parse(content)
        except Exception:
            logger.debug("openai_json_mode_failed", model=self.model)
        return super().complete_json(system_prompt, user_prompt)
