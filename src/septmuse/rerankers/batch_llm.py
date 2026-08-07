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
"""LLM 批量打分 reranker — 一次请求对多个文档打分, 减少 API 调用。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger
from septmuse.rerankers.base import BaseReranker
from septmuse.rerankers.llm import LLMReranker

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)


class BatchLLMReranker(BaseReranker):
    """LLM 批量打分, 解析失败回退到逐条打分。"""

    _MAX_INPUT_LEN = 2000
    _MAX_BATCH_SIZE = 10
    _SYSTEM_PROMPT = (
        "You are a relevance scoring assistant. "
        "Given a query and multiple documents, score how relevant each document is to the query.\n\n"
        "Score the relevance on a scale from 0.0 to 1.0, where:\n"
        "- 1.0 = Perfectly relevant and directly answers the query\n"
        "- 0.8-0.9 = Highly relevant with good information\n"
        "- 0.6-0.7 = Moderately relevant with some useful information\n"
        "- 0.4-0.5 = Slightly relevant with limited useful information\n"
        "- 0.0-0.3 = Not relevant or no useful information\n\n"
        "Respond with ONLY a JSON array of objects with 'id' (integer) and 'score' (float) fields.\n"
        'Example: [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.3}]'
    )

    def __init__(self, llm: LLM | None = None, **kwargs) -> None:
        self._llm = llm
        self._fallback = LLMReranker(llm=llm)

    def _parse_batch_response(self, response_text: str, count: int) -> list[float] | None:
        try:
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start < 0 or end < 0:
                return None
            items = json.loads(response_text[start : end + 1])
            scores = [0.5] * count
            for item in items:
                idx = item.get("id")
                score = item.get("score", 0.5)
                if isinstance(idx, int) and 0 <= idx < count:
                    scores[idx] = min(max(float(score), 0.0), 1.0)
            return scores
        except Exception:
            return None

    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None) -> list[tuple[int, float]]:
        if not documents:
            return []
        if self._llm is None:
            raise ValueError("BatchLLMReranker requires an LLM instance")
        all_scored: list[tuple[int, float]] = []
        offset = 0
        for i in range(0, len(documents), self._MAX_BATCH_SIZE):
            batch = documents[i : i + self._MAX_BATCH_SIZE]
            batch_scored = self._rerank_batch(query, batch, offset)
            all_scored.extend(batch_scored)
            offset += len(batch)
        all_scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            all_scored = all_scored[:top_k]
        return all_scored

    def _rerank_batch(self, query: str, batch: list[str], offset: int) -> list[tuple[int, float]]:
        safe_query = query[: self._MAX_INPUT_LEN]
        doc_lines = [f"[{i}] {doc[: self._MAX_INPUT_LEN]}" for i, doc in enumerate(batch)]
        user_prompt = f"Query: {safe_query}\n\nDocuments:\n" + "\n".join(doc_lines)
        try:
            response = self._llm.complete(self._SYSTEM_PROMPT, user_prompt)
            scores = self._parse_batch_response(response, len(batch))
        except Exception as e:
            logger.warning("batch_llm_rerank_failed", error=str(e), fallback="single")
            scores = None
        if scores is None:
            fallback_scored = self._fallback.rerank(query, batch, top_k=len(batch))
            return [(offset + idx, score) for idx, score in fallback_scored]
        return [(offset + i, score) for i, score in enumerate(scores)]
