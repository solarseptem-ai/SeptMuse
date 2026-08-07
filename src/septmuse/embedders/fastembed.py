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
"""FastEmbed 嵌入 provider — 轻量 ONNX, 无 torch。

与 OnnxEmbedder 功能重叠, 但使用 fastembed 库 (不同模型生态)。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "thenlper/gte-large"


class FastEmbedEmbedder(Embedder):
    """FastEmbed Embeddings provider (轻量 ONNX)。"""

    def __init__(self, model: str = DEFAULT_MODEL, embedding_dims: int | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("fastembed required: pip install septmuse[fastembed]") from e

        self.backend_name = "fastembed"
        self.model = model
        logger.info("embedder_loading", provider="fastembed", model=model)
        self._model = TextEmbedding(model_name=model)
        self._dim = embedding_dims or self._model.embedding_size
        logger.info("embedder_ready", provider="fastembed", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        embeddings = list(self._model.embed(text))
        return embeddings[0]

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ") for t in texts]
        results = list(self._model.embed(cleaned))
        return results
