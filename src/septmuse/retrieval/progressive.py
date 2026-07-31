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
"""渐进式三层检索 — ReMe chunk→file→link 三层模式适配 SeptMuse。

借鉴 ReMe 渐进三层 (源码实证 reme/steps/index/):
- Layer 1 recall: chunk 召回 (向量检索 verbatim + typed memories)
- Layer 2 locate: 定位来源 (标记 memory_type + 溯源 metadata)
- Layer 3 expand: 链接邻居 (同 user_id + tags 关联 + 时序邻近)

ReMe 实际三层是 chunk→file location→link neighbors, 非设计的 meta→vector→history。
本模块适配为 SeptMuse 的 recall→locate→expand 三层。

详见 docs/specs/agent-memory-architecture.md §5.2 检索策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.storage.base import MemoryStore
from septmuse.storage.typed_store import TypedMemoryStore

logger = get_logger(__name__)

# Layer 3 expand 默认邻居数
DEFAULT_EXPAND_LIMIT = 3


@dataclass
class ProgressiveResult:
    """渐进检索结果项。"""

    id: str
    memory: str
    score: float
    memory_type: str = "verbatim"  # verbatim | semantic | episodic | procedural
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    expanded_from: str | None = None  # Layer 3: 从哪条记忆扩展而来


class ProgressiveRetriever:
    """渐进式三层检索 (对齐 ReMe recall→locate→expand 模式)。

    用法:
        retriever = ProgressiveRetriever(store, typed_store, embedder)
        results = retriever.retrieve("alice likes python", user_id="alice")
        # results 含 Layer1 recall + Layer3 expand 的合并去重结果
    """

    def __init__(
        self,
        store: MemoryStore,
        typed_store: TypedMemoryStore,
        embedder: Embedder,
        expand_limit: int = DEFAULT_EXPAND_LIMIT,
    ) -> None:
        self.store = store
        self.typed_store = typed_store
        self.embedder = embedder
        self.expand_limit = expand_limit

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[ProgressiveResult]:
        """三层渐进检索。

        Layer 1: recall — 向量检索 verbatim + typed memories
        Layer 2: locate — 标记 memory_type, 溯源 metadata
        Layer 3: expand — 按 tags/user_id 时序邻近扩展邻居
        """
        # Layer 1 + 2: recall + locate
        recalled = self._recall_and_locate(query, user_id=user_id, top_k=top_k, threshold=threshold)
        if not recalled:
            return []

        # Layer 3: expand — 从召回结果扩展邻居
        expanded = self._expand(recalled, user_id=user_id)

        # 合并 + 去重 (by id)
        seen: set[str] = set()
        merged: list[ProgressiveResult] = []
        for r in recalled + expanded:
            if r.id not in seen:
                seen.add(r.id)
                merged.append(r)

        # 按 score 降序
        merged.sort(key=lambda x: x.score, reverse=True)
        logger.info("progressive_retrieve", recalled=len(recalled), expanded=len(expanded), merged=len(merged))
        return merged[:top_k]

    def _recall_and_locate(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
        threshold: float,
    ) -> list[ProgressiveResult]:
        """Layer 1+2: 向量召回 + 标记来源类型。"""
        results: list[ProgressiveResult] = []

        # verbatim memories (SQLiteMemoryStore)
        emb = self.embedder.embed(query)
        for r in self.store.search(emb, user_id=user_id, top_k=top_k, threshold=threshold):
            results.append(
                ProgressiveResult(
                    id=r["id"],
                    memory=r["memory"],
                    score=r["score"],
                    memory_type="verbatim",
                    metadata=r.get("metadata", {}),
                    created_at=r.get("created_at"),
                )
            )

        # semantic facts (TypedMemoryStore, 置信度加权)
        facts = self.typed_store.search_facts(emb, user_id=user_id, top_k=top_k)
        for f in facts:
            results.append(
                ProgressiveResult(
                    id=str(f["id"]),
                    memory=f"{f['subject']} {f['predicate']} {f['object']}",
                    score=f["score"] * f.get("confidence", 1.0),
                    memory_type="semantic",
                    metadata={
                        "triple": {
                            "subject": f["subject"],
                            "predicate": f["predicate"],
                            "object": f["object"],
                            "context": f.get("context"),
                        },
                        "confidence": f.get("confidence", 1.0),
                        "provenance": f.get("provenance", "user"),
                    },
                )
            )

        return results

    def _expand(self, recalled: list[ProgressiveResult], *, user_id: str) -> list[ProgressiveResult]:
        """Layer 3: 链接邻居 — 按 tags 关联 + 时序邻近扩展。

        简化版 (无图结构): 从 recalled 的 tags metadata 找同标签记忆,
        或从同 user_id 取时序邻近记忆 (ReMe link expansion 模式适配)。
        """
        if not recalled:
            return []

        expanded: list[ProgressiveResult] = []
        seen_ids = {r.id for r in recalled}

        # 从 recalled 的 tags 找关联
        all_memories = self.store.get_all(user_id=user_id)
        for source in recalled[: self.expand_limit]:
            source_tags = source.metadata.get("tags", [])
            for mem in all_memories:
                if mem["id"] in seen_ids:
                    continue
                mem_tags = mem.get("metadata", {}).get("tags", [])
                # tag 交集 → 邻居
                if source_tags and mem_tags and set(source_tags) & set(mem_tags):
                    expanded.append(
                        ProgressiveResult(
                            id=mem["id"],
                            memory=mem["memory"],
                            score=source.score * 0.5,  # 扩展分数衰减
                            memory_type="verbatim",
                            metadata=mem.get("metadata", {}),
                            created_at=mem.get("created_at"),
                            expanded_from=source.id,
                        )
                    )
                    seen_ids.add(mem["id"])
                    if len(expanded) >= self.expand_limit * 2:
                        break

        return expanded
