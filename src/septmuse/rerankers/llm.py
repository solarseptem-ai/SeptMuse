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
"""LLM 逐条打分 reranker — LLM.complete() 对每个文档打分 0-1。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger
from septmuse.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)


class LLMReranker(BaseReranker):
    """LLM 逐条打分 reranker, 失败回退中性分 0.5。"""

    _MAX_INPUT_LEN = 4000
    _SYSTEM_PROMPT = (
        "You are a relevance scoring assistant. "
        "Given a query and a document, score how relevant the document is to the query.\n\n"
        "Score the relevance on a scale from 0.0 to 1.0.\n"
        "Respond with only a single numerical score between 0.0 and 1.0."
    )

    def __init__(self, llm: LLM | None = None, **kwargs) -> None:
        self._llm = llm

    def _extract_score(self, response_text: str) -> float:
        matches = re.findall(r"-?\d+\.\d+", response_text) or re.findall(r"-?\d+", response_text)
        if matches:
            return min(max(float(matches[0]), 0.0), 1.0)
        return 0.5

    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None) -> list[tuple[int, float]]:
        if not documents:
            return []
        if self._llm is None:
            raise ValueError("LLMReranker requires an LLM instance")
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            safe_query = query[: self._MAX_INPUT_LEN]
            safe_doc = doc[: self._MAX_INPUT_LEN]
            user_prompt = f"Query: {safe_query}\n\nDocument: {safe_doc}"
            try:
                response = self._llm.complete(self._SYSTEM_PROMPT, user_prompt)
                score = self._extract_score(response)
            except Exception as e:
                logger.warning("llm_rerank_failed", doc_index=i, error=str(e))
                score = 0.5
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return scored
