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
"""透传 reranker — 不改变顺序和 score。"""

from __future__ import annotations

from septmuse.rerankers.base import BaseReranker


class NoopReranker(BaseReranker):
    """透传 reranker, 不打分, 返回原始顺序。"""

    def __init__(self, **kwargs) -> None:
        pass

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        limit = top_k or len(documents)
        return [(i, 0.0) for i in range(min(limit, len(documents)))]
