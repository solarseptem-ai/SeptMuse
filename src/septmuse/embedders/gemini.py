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
"""Google Gemini 嵌入 provider — google-genai SDK。"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "models/gemini-embedding-001"
DEFAULT_DIMS = 768
MAX_BATCH = 100


class GeminiEmbedder(Embedder):
    """Google Gemini Embeddings provider。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        embedding_dims: int = DEFAULT_DIMS,
        output_dimensionality: int | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("google-genai package required: pip install septmuse[gemini]") from e

        self.backend_name = "gemini"
        self.model = model
        self._dim = embedding_dims
        self._output_dimensionality = output_dimensionality or embedding_dims
        self._types = types

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("Google API key required: set GOOGLE_API_KEY or pass api_key=")

        logger.info("embedder_loading", provider="gemini", model=model, dim=self._dim)
        self._client = genai.Client(api_key=resolved_key)
        logger.info("embedder_ready", provider="gemini", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        config = self._types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        response = self._client.models.embed_content(model=self.model, contents=text, config=config)
        return response.embeddings[0].values

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        config = self._types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = [t.replace("\n", " ") for t in texts[i : i + MAX_BATCH]]
            response = self._client.models.embed_content(model=self.model, contents=chunk, config=config)
            all_embeddings.extend(e.values for e in response.embeddings)
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"Gemini embed_batch() returned {len(all_embeddings)} embeddings for {len(texts)} texts"
            )
        return all_embeddings
