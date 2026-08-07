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
"""SentenceTransformer cross-encoder reranker (借鉴 mem0 SentenceTransformerReranker)。

模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (默认, sentence-transformers 自带)。
推理: (query, document) pair -> CrossEncoder.predict -> logit -> sigmoid 归一化。
降级: sentence-transformers 不可用时降级为透传 + warning。

与 CrossEncoderReranker (ONNX) 的区别:
- 本类用 sentence-transformers 库 (需 torch, 启动慢 ~30s, 模型质量高)
- CrossEncoderReranker 用 ONNX (无 torch, 轻量, 量化版模型)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.rerankers.base import BaseReranker

logger = get_logger(__name__)

_DEFAULT_ST_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SentenceTransformerReranker(BaseReranker):
    """sentence-transformers CrossEncoder reranker。

    延迟 import, 不可用降级为透传。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_ST_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._device = device  # None=自动检测
        self._batch_size = batch_size
        self._show_progress_bar = show_progress_bar
        self._normalize = normalize
        self._model: Any = None
        self._degraded = False
        self._init_attempted = False

    def _init_model(self) -> None:
        """延迟加载 sentence-transformers CrossEncoder。"""
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("st_reranker_degraded", reason="sentence-transformers not installed")
            self._degraded = True
            return
        try:
            logger.info("st_reranker_loading", model=self._model_name)
            self._model = CrossEncoder(self._model_name, device=self._device)
            logger.info("st_reranker_ready", model=self._model_name)
        except Exception as e:
            logger.warning("st_reranker_degraded", reason=str(e))
            self._degraded = True

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        self._init_model()
        if self._degraded or self._model is None:
            limit = top_k or len(documents)
            return [(i, 0.5) for i in range(min(limit, len(documents)))]

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=self._show_progress_bar,
            )
            scores = np.asarray(scores, dtype=np.float64)
            if self._normalize:
                scores = 1.0 / (1.0 + np.exp(-scores))
            scored = [(i, float(s)) for i, s in enumerate(scores)]
        except Exception as e:
            logger.warning("st_rerank_failed", error=str(e), fallback="neutral")
            scored = [(i, 0.5) for i in range(len(documents))]

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return scored
