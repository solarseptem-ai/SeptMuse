#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Cohere 云 reranker — 通过 Cohere API 对文档重排。

延迟 import cohere, 不可用时降级为透传 + warning。
需要 COHERE_API_KEY 环境变量或构造参数传入。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.rerankers.base import BaseReranker

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    """Cohere reranker — 云端 API 打分。"""

    def __init__(self, api_key: str | None = None, model: str = "rerank-v3.5", **kwargs) -> None:
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._model = model
        self._client: Any = None
        self._degraded = False
        self._init_attempted = False

    def _init_client(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        if not self._api_key:
            logger.warning("cohere_reranker_degraded", reason="COHERE_API_KEY not set")
            self._degraded = True
            return
        try:
            import cohere
            self._client = cohere.Client(self._api_key)
            logger.info("cohere_reranker_ready", model=self._model)
        except ImportError:
            logger.warning("cohere_reranker_degraded", reason="cohere package not installed")
            self._degraded = True
        except Exception as e:
            logger.warning("cohere_reranker_degraded", reason=str(e))
            self._degraded = True

    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None) -> list[tuple[int, float]]:
        if not documents:
            return []
        self._init_client()
        if self._degraded or self._client is None:
            limit = top_k or len(documents)
            return [(i, 0.5) for i in range(min(limit, len(documents)))]
        try:
            response = self._client.rerank(
                model=self._model, query=query, documents=documents, top_n=top_k or len(documents)
            )
            scored = [(item.index, float(item.relevance_score)) for item in response.results]
            return scored
        except Exception as e:
            logger.warning("cohere_rerank_failed", error=str(e), fallback="original_order")
            limit = top_k or len(documents)
            return [(i, 0.5) for i in range(min(limit, len(documents)))]
