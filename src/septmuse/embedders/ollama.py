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
"""Ollama 嵌入 provider — 本地 Ollama 嵌入, 自动 pull 模型。

零 API key, 本地 Ollama 服务。首次使用自动 pull 模型。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIMS = 512


class OllamaEmbedder(Embedder):
    """基于 Ollama 的本地嵌入。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        embedding_dims: int = DEFAULT_DIMS,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.backend_name = "ollama"
        self.model = model
        self._dim = embedding_dims
        self._client = Client(host=ollama_base_url)
        self._ensure_model_exists()
        logger.info("ollama_embedder_ready", model=model, dim=self._dim)

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    def _ensure_model_exists(self) -> None:
        local_models = self._client.list()["models"]
        target = self._normalize_model_name(self.model)
        if not any(
            self._normalize_model_name(m.get("name", "")) == target
            or self._normalize_model_name(m.get("model", "")) == target
            for m in local_models
        ):
            logger.info("ollama_model_pulling", model=self.model)
            self._client.pull(self.model)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        response = self._client.embed(model=self.model, input=text)
        embeddings = response.get("embeddings") or []
        if not embeddings:
            raise ValueError(f"Ollama embed() returned no embeddings for model '{self.model}'")
        return embeddings[0]

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(model=self.model, input=texts)
        embeddings = response.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama embed() returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        return embeddings
