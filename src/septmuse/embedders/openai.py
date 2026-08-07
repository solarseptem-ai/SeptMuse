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
"""OpenAI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

用法:
    embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")
    vec = embedder.embed("hello")

零配置: 从环境变量 OPENAI_API_KEY 读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 1536


class OpenAIEmbedder(_OpenAICompatibleEmbedder):
    """OpenAI Embeddings provider。

    零配置: 从 OPENAI_API_KEY 环境变量读取。
    自定义: OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")。

    matryoshka 支持: 显式传 embedding_dims 时向 API 传 dimensions 参数
    (兼容非 matryoshka 后端如 vLLM/Voyage, 它们拒绝 dimensions 参数)。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        embedding_dims: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS
        pass_dimensions = embedding_dims is not None

        self._api_key = api_key or os.getenv("SEPTMUSE_EMBEDDER_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-required"

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("SEPTMUSE_BASE_URL") or os.getenv("SEPTMUSE_EMBEDDER_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        client_kwargs.update(kwargs)

        logger.info("embedder_loading", model=model, dim=dim)
        client = OpenAI(**client_kwargs)
        logger.info("embedder_ready", model=model, dim=dim)

        super().__init__(
            client=client,
            model=model,
            dim=dim,
            pass_dimensions_to_api=pass_dimensions,
        )
        self.backend_name = "openai"
