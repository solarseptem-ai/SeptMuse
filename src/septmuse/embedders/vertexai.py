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
"""Vertex AI 嵌入 provider — 唯一真正用 memory_action 的 provider。

memory_action 切换 task_type:
- "add"/"update" → RETRIEVAL_DOCUMENT
- "search" → RETRIEVAL_QUERY
- None → SEMANTIC_SIMILARITY
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMS = 256
BATCH_SIZE = 250


class VertexAIEmbedder(Embedder):
    """Google Vertex AI Embeddings provider (memory_action task_type 切换)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_dims: int = DEFAULT_DIMS,
        vertex_credentials_json: str | None = None,
        memory_add_embedding_type: str = "RETRIEVAL_DOCUMENT",
        memory_search_embedding_type: str = "RETRIEVAL_QUERY",
        memory_update_embedding_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        try:
            from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
        except ImportError as e:
            raise ImportError("google-cloud-aiplatform required: pip install septmuse[vertexai]") from e

        self.backend_name = "vertexai"
        self.model = model
        self._dim = embedding_dims
        self._TextEmbeddingInput = TextEmbeddingInput

        self._embedding_types = {
            "add": memory_add_embedding_type,
            "update": memory_update_embedding_type,
            "search": memory_search_embedding_type,
        }

        creds = vertex_credentials_json or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

        logger.info("embedder_loading", provider="vertexai", model=model, dim=self._dim)
        self._model = TextEmbeddingModel.from_pretrained(model)
        logger.info("embedder_ready", provider="vertexai", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _resolve_task_type(self, memory_action: str | None) -> str:
        if memory_action is None:
            return "SEMANTIC_SIMILARITY"
        if memory_action not in self._embedding_types:
            raise ValueError(f"Invalid memory_action: {memory_action}")
        return self._embedding_types[memory_action]

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        task_type = self._resolve_task_type(memory_action)
        text_input = self._TextEmbeddingInput(text=text, task_type=task_type)
        embeddings = self._model.get_embeddings(texts=[text_input], output_dimensionality=self._dim)
        return embeddings[0].values

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        task_type = self._resolve_task_type(memory_action)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            inputs = [self._TextEmbeddingInput(text=t, task_type=task_type) for t in chunk]
            results = self._model.get_embeddings(texts=inputs, output_dimensionality=self._dim)
            all_embeddings.extend(r.values for r in results)
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"Vertex AI embed_batch() returned {len(all_embeddings)} embeddings for {len(texts)} texts"
            )
        return all_embeddings
