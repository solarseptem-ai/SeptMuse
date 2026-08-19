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
"""OpenAI 兼容嵌入基类。"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder

MAX_BATCH = 100


class _OpenAICompatibleEmbedder(Embedder):
    """OpenAI 兼容嵌入基类。"""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dim: int,
        pass_dimensions_to_api: bool,
    ) -> None:
        self._client = client
        self._model = model
        self._dim = dim
        self._pass_dimensions_to_api = pass_dimensions_to_api
        self.backend_name = "openai_compatible"

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        kwargs: dict[str, Any] = {
            "input": [text],
            "model": self._model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self._dim
        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(cleaned), MAX_BATCH):
            chunk = cleaned[i : i + MAX_BATCH]
            kwargs: dict[str, Any] = {
                "input": chunk,
                "model": self._model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self._dim
            response = self._client.embeddings.create(**kwargs)
            all_embeddings.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )

        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"embed_batch() returned {len(all_embeddings)} embeddings "
                f"for {len(texts)} texts using model '{self._model}'"
            )
        return all_embeddings
