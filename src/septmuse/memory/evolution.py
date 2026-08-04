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
"""V2 演化子组件 — 聚合 Dream 链接生长 + reflect 蒸馏 + 冲突解决。

EvolutionEngine 聚合 evolution/ 下三子模块:
- DreamIntegrator: 空闲期批量建立记忆间链接 (embedding 相似度)
- SessionReflector: 从情节推理经验蒸馏程序规则 (LLM 或 heuristic)
- ConflictResolver: 矛盾事实检测 + 实体去重

详见 docs/specs/2026-08-04-v2-memory-architecture.md §4 + §6。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.evolution.conflict import ConflictResolver
from septmuse.evolution.dream import DreamIntegrator, DreamResult
from septmuse.evolution.reflect import SessionReflector
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class EvolutionEngine:
    """V2 演化引擎 — 聚合 Dream + reflect + 冲突。

    用法:
        engine = EvolutionEngine(store, graph_store, embedder, typed_store, llm)
        dream = engine.dream(user_id="alice")
        rules = engine.reflect(user_id="alice")
        conflicts = engine.resolve_conflicts(user_id="alice")
    """

    def __init__(
        self,
        store: MemoryStore,
        graph_store: GraphStore | None,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
    ) -> None:
        self.dreamer = DreamIntegrator(store, graph_store, embedder)
        self.reflector = SessionReflector(typed_store, llm=llm)
        self.conflict_resolver = ConflictResolver(typed_store, store, llm)

    def dream(self, *, user_id: str) -> DreamResult:
        """Dream 链接生长 (空闲期批量建链接)。"""
        return self.dreamer.dream(user_id=user_id)

    def reflect(self, *, user_id: str, limit: int = 20) -> Any:
        """情节 → 程序规则蒸馏 (LLM 或 heuristic)。"""
        return self.reflector.reflect(user_id=user_id, limit=limit)

    def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """矛盾事实检测 + 解决 (新覆盖旧, 软删除旧事实)。"""
        return self.conflict_resolver.resolve_conflicts(user_id=user_id)

    def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """实体去重三段式 (精确归一化 + 模糊相似度 + LLM 兜底)。"""
        return self.conflict_resolver.deduplicate_entities(user_id=user_id)
