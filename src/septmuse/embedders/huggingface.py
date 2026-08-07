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
"""HuggingFace 嵌入 provider — 双模式。

有 huggingface_base_url 时走 TEI server (OpenAI 兼容 API);
无 base_url 时走本地 SentenceTransformer。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"


class HuggingFaceEmbedder(Embedder):
    """HuggingFace 嵌入 (TEI server 或 本地 SentenceTransformer)。"""

    def __init__(
        self,
        model: str | None = None,
        huggingface_base_url: str | None = None,
        model_kwargs: dict | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        self.backend_name = "huggingface"
        self._tei_mode = huggingface_base_url is not None
        self._kwargs = model_kwargs or {}

        if self._tei_mode:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai package required for TEI mode: pip install septmuse[openai]") from e

            resolved_model = model or "tei"
            logger.info("embedder_loading", provider="huggingface_tei", model=resolved_model, base_url=huggingface_base_url)
            client = OpenAI(base_url=huggingface_base_url)
            dim = embedding_dims or 768
            logger.info("embedder_ready", provider="huggingface_tei", model=resolved_model, dim=dim)

            self._inner: Embedder = _OpenAICompatibleEmbedder(
                client=client, model=resolved_model, dim=dim, pass_dimensions_to_api=False
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("sentence-transformers required: pip install septmuse[st]") from e

            resolved_model = model or DEFAULT_MODEL
            logger.info("embedder_loading", provider="huggingface_local", model=resolved_model)
            self._st_model = SentenceTransformer(resolved_model, **self._kwargs)
            dim = embedding_dims or self._st_model.get_sentence_embedding_dimension()
            assert dim is not None
            self._dim: int = dim
            logger.info("embedder_ready", provider="huggingface_local", model=resolved_model, dim=dim)

            self._inner = _LocalSTAdapter(self._st_model, self._dim)

        self.model = resolved_model

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._inner._embed(text, memory_action)

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return self._inner._embed_batch(texts, memory_action)


class _LocalSTAdapter(Embedder):
    """SentenceTransformer 适配器 — 包装 ST 模型为 Embedder 接口。"""

    def __init__(self, model: Any, dim: int) -> None:
        self.backend_name = "huggingface_local"
        self._model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
