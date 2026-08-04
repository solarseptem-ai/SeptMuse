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
"""语义记忆操作 — 三元组 CRUD + 抽取流水线占位。

阶段2: 提供 SemanticStore 包装 TypedMemoryStore 的语义事实 CRUD。
LLM 抽取流水线 (cognify) 见 extract.py (阶段2 后续)。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.models.semantic import SemanticFact
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class SemanticMemory:
    """语义记忆操作 (架构文档 §3.2.2)。

    包装 TypedMemoryStore, 提供语义事实 CRUD + 向量检索。
    身份子类通过 tags=["identity"] 复用同一表 (不单独立层)。
    """

    def __init__(self, store: TypedMemoryStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        user_id: str,
        context: str | None = None,
        confidence: float = 1.0,
        provenance: str = "user",
        tags: list[str] | None = None,
        embed: bool = True,
    ) -> SemanticFact:
        """添加语义事实。embed=True 时自动生成向量 (供语义检索)。"""
        embedding = None
        if embed:
            text = f"{subject} {predicate} {object}"
            if context:
                text += f" {context}"
            embedding = self.embedder.embed(text)
        return self.store.add_fact(
            subject,
            predicate,
            object,
            user_id=user_id,
            context=context,
            confidence=confidence,
            provenance=provenance,
            tags=tags,
            embedding=embedding,
        )

    def search_facts(self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1) -> list[dict[str, Any]]:
        """语义检索事实 (向量 + 置信度加权, SeptMuse 增量)。"""
        emb = self.embedder.embed(query)
        results = self.store.search_facts(emb, user_id=user_id, top_k=top_k, threshold=threshold)
        # SeptMuse 增量: 置信度加权排序
        for r in results:
            r["final_score"] = r["score"] * r.get("confidence", 1.0)
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def get_all_facts(self, *, user_id: str) -> list[SemanticFact]:
        return self.store.get_all_facts(user_id=user_id)

    def add_identity(self, key: str, value: str, *, user_id: str) -> SemanticFact:
        """添加身份记忆子类 (架构文档 §3.2.2: 身份归语义, 打 identity 标签)。"""
        return self.add_fact(
            subject=user_id,
            predicate="identity",
            object=f"{key}: {value}",
            user_id=user_id,
            tags=["identity"],
            confidence=1.0,
            provenance="user",
        )
