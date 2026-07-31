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
"""OpenAI 嵌入 provider (借鉴 mem0 embeddings/openai.py OpenAIEmbedding 模式)。

对齐 septmuse.embedders.base.Embedder ABC,
调用 OpenAI Embeddings API。

用法:
    embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")
    vec = embedder.embed("hello")
    vecs = embedder.embed_batch(["hello", "world"])

零配置: 从环境变量 OPENAI_API_KEY 读取 key (对齐 providers/llms/openai.py)。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 1536
MAX_BATCH = 100


class OpenAIEmbedder(Embedder):
    """OpenAI Embeddings provider (借鉴 mem0 OpenAIEmbedding)。

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
        self._dim = embedding_dims or DEFAULT_DIMS
        # 仅当用户显式设了 embedding_dims 才向 API 传 dimensions 参数
        # (非 matryoshka 后端如 vLLM/Voyage 拒绝 dimensions 参数, 对齐 mem0)
        self._pass_dimensions_to_api = embedding_dims is not None

        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-required"

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("SEPTMUSE_EMBEDDER_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        client_kwargs.update(kwargs)

        logger.info("embedder_loading", model=model, dim=self._dim)
        self._client = OpenAI(**client_kwargs)
        logger.info("embedder_ready", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """嵌入单条文本 (对齐 Embedder ABC + mem0 OpenAIEmbedding.embed)。

        换行替换为空格 (OpenAI 要求, 对齐 mem0)。
        """
        text = text.replace("\n", " ")
        kwargs: dict[str, Any] = {
            "input": [text],
            "model": self.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self._dim
        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入 (对齐 mem0 OpenAIEmbedding.embed_batch)。

        100 一批分块, 按 .index 排序, 数量校验。
        """
        if not texts:
            return []

        cleaned = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(cleaned), MAX_BATCH):
            chunk = cleaned[i : i + MAX_BATCH]
            kwargs: dict[str, Any] = {
                "input": chunk,
                "model": self.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self._dim
            response = self._client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))

        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"OpenAI embed_batch() returned {len(all_embeddings)} embeddings "
                f"for {len(texts)} texts using model '{self.model}'"
            )
        return all_embeddings
