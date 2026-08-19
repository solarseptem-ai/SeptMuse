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
"""Zettelkasten 链接生长 — add 时自动找语义关系建链接。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore

logger = get_logger(__name__)

DEFAULT_LINK_THRESHOLD = 0.3
DEFAULT_MAX_LINKS = 5


@dataclass
class MemoryLink:
    """记忆间链接。"""

    id: str
    source_id: str
    target_id: str
    relation: str
    score: float


class ZettelLinker:
    """Zettelkasten 链接生长器。"""

    def __init__(
        self,
        store: MemoryStore,
        graph_store: GraphStore,
        embedder: Embedder,
        threshold: float = DEFAULT_LINK_THRESHOLD,
        max_links: int = DEFAULT_MAX_LINKS,
    ) -> None:
        self.store = store
        self.graph_store = graph_store
        self.embedder = embedder
        self.threshold = threshold
        self.max_links = max_links

    def link_on_add(
        self,
        memory_id: str,
        text: str,
        embedding: list[float],
        *,
        user_id: str,
        relation: str = "related_to",
    ) -> list[MemoryLink]:
        """add 后自动找链接。"""
        candidates = self.store.search(
            embedding,
            user_id=user_id,
            top_k=self.max_links * 2,
            threshold=self.threshold,
        )

        existing = set(self.graph_store.get_neighbors(memory_id, relation))
        links: list[MemoryLink] = []
        for c in candidates:
            target_id = c["id"]
            if target_id == memory_id or target_id in existing:
                continue
            if len(links) >= self.max_links:
                break
            edge1_id = self.graph_store.add_edge(memory_id, target_id, relation, c["score"])
            self.graph_store.add_edge(target_id, memory_id, relation, c["score"])
            links.append(
                MemoryLink(
                    id=edge1_id,
                    source_id=memory_id,
                    target_id=target_id,
                    relation=relation,
                    score=c["score"],
                )
            )

        logger.info("zettel_link_done", memory_id=memory_id, links_created=len(links))
        return links

    def get_links(self, memory_id: str) -> list[MemoryLink]:
        """获取记忆的所有链接 (转换 GraphEdge → MemoryLink)。"""
        return [
            MemoryLink(id=e.id, source_id=e.source_id, target_id=e.target_id, relation=e.relation, score=e.score)
            for e in self.graph_store.get_edges(memory_id)
        ]

    def get_related_memories(self, memory_id: str) -> list[dict[str, Any]]:
        """获取链接的记忆内容 (含 metadata)。"""
        links = self.get_links(memory_id)
        results: list[dict[str, Any]] = []
        for link in links:
            mem = self.store.get(link.target_id)
            if mem:
                results.append({**mem, "link_score": link.score, "link_relation": link.relation})
        return results
