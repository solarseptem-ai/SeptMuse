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
"""Dream 整合 — 空闲期批量建立记忆间链接 (借鉴 ReMe Dream 4-phase)。

借鉴 (源码实证):
- ReMe Dream 4-phase:
  1. DreamExtractStep: 扫描 daily notes, 提取 units/topics
  2. DreamIntegrateStep: 对每个 unit, node_search 找关联, write/edit 织入 wikilink
     (IntegrateOutcome: CREATE/CORROBORATE/REFINE/CORRECT)
  3. DreamTopicsStep: 写 interests.yaml
  4. DreamFinishStep: 持久化 catalog
- ReMe DreamBucketEnum: procedure/personal/wiki

SeptMuse 简化:
- 不用 LLM agent + 工具调用, 用 embedding 相似度批量建链接
- 2-phase: extract (聚类) → integrate (建链接)
- 复用 ZettelLinker 的 link 机制

详见 docs/specs/agent-memory-architecture.md §5.4 演化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.evolution.zettel import ZettelLinker
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph.base import GraphStore

logger = get_logger(__name__)

# 默认批量大小
DEFAULT_BATCH_SIZE = 50


@dataclass
class DreamResult:
    """Dream 整合结果。"""

    processed: int = 0
    links_created: int = 0
    clusters_found: int = 0
    errors: list[str] = field(default_factory=list)


class DreamIntegrator:
    """Dream 整合器 (对齐 ReMe Dream: extract → integrate 2-phase 简化版)。

    空闲期批量处理所有记忆, 为每条记忆找语义关联并建立链接。

    用法:
        dreamer = DreamIntegrator(store, embedder)
        result = dreamer.dream(user_id="alice")
        # result.links_created = N
    """

    def __init__(
        self,
        store: MemoryStore,
        graph_store: GraphStore,
        embedder: Embedder,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threshold: float = 0.3,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.batch_size = batch_size
        self.linker = ZettelLinker(store, graph_store, embedder, threshold=threshold)

    def dream(self, *, user_id: str) -> DreamResult:
        """执行 Dream 整合 (对齐 ReMe Dream extract → integrate)。

        Phase 1 extract: 取全部记忆, 重新嵌入
        Phase 2 integrate: 对每条记忆, 找语义关联, 建链接 (复用 ZettelLinker)
        """
        result = DreamResult()

        # Phase 1: extract — 取全部记忆
        all_memories = self.store.get_all(user_id=user_id)
        if not all_memories:
            logger.info("dream_no_memories", user_id=user_id)
            return result

        result.processed = min(len(all_memories), self.batch_size)

        # Phase 2: integrate — 批量建链接
        for mem in all_memories[: self.batch_size]:
            try:
                text = mem["memory"]
                mem_id = mem["id"]
                emb = self.embedder.embed(text)
                links = self.linker.link_on_add(mem_id, text, emb, user_id=user_id)
                result.links_created += len(links)
            except Exception as e:
                result.errors.append(f"{mem.get('id', '?')}: {e}")

        logger.info(
            "dream_done",
            user_id=user_id,
            processed=result.processed,
            links_created=result.links_created,
            errors=len(result.errors),
        )
        return result

    def get_clusters(self, *, user_id: str, threshold: float = 0.5) -> list[list[dict[str, Any]]]:
        """获取记忆聚类 (相似度 > threshold 的记忆组)。

        简化聚类: 用链接关系做连通分量 (union-find)。
        """
        all_memories = self.store.get_all(user_id=user_id)
        if not all_memories:
            return []

        mem_ids = {m["id"] for m in all_memories}
        # union-find
        parent: dict[str, str] = {mid: mid for mid in mem_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for m in all_memories:
            links = self.linker.get_links(m["id"])
            for link in links:
                if link.target_id in mem_ids:
                    union(m["id"], link.target_id)

        # 按根分组
        clusters: dict[str, list[dict[str, Any]]] = {}
        for m in all_memories:
            root = find(m["id"])
            clusters.setdefault(root, []).append(m)

        # 只返回 >1 条记忆的聚类
        return [c for c in clusters.values() if len(c) > 1]
