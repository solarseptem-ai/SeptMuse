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
"""Reranker 抽象基类 — 对纯文本文档打分。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """重排器抽象基类 — 对纯文本文档打分。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对 documents 打分，返回按相关性降序的 [(doc_index, score)]。"""
        ...
