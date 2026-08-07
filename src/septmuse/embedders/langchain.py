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
"""Langchain 嵌入 provider — 桥接 LangChain Embeddings。

接收用户传入的 langchain.embeddings.Embeddings 实例,
调 embed_query() / embed_documents()。一次桥接整个 LangChain embedding 生态。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)


class LangchainEmbedder(Embedder):
    """LangChain Embeddings 桥接器。"""

    def __init__(self, model: Any, embedding_dims: int = 768) -> None:
        self.backend_name = "langchain"
        if model is None:
            raise ValueError("`model` parameter is required (langchain Embeddings instance)")

        if not (hasattr(model, "embed_query") and hasattr(model, "embed_documents")):
            try:
                from langchain.embeddings.base import Embeddings
            except ImportError:
                try:
                    from langchain_core.embeddings import Embeddings
                except ImportError as e:
                    raise ImportError("langchain required: pip install septmuse[langchain]") from e
            if not isinstance(model, Embeddings):
                raise ValueError("`model` must be an instance of langchain Embeddings")

        self._langchain_model = model
        self._dim = embedding_dims
        logger.info("embedder_ready", provider="langchain", dim=self._dim, type=type(model).__name__)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._langchain_model.embed_query(text)

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        return self._langchain_model.embed_documents(texts)
