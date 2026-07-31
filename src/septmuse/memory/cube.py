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
"""MemCube 统一容器 (架构文档 §7.2 自研, 借鉴 MemOS MemCube)。

MemCube 是多类型记忆的统一容器, 封装 Memory facade + 注册表,
提供类型路由: 根据 memory_type 分发到对应 handler。

注: cognee 无 MemCube 概念 (探查确认), 本模块为架构文档自研设计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.memory.main import Memory
from septmuse.memory.registry import MemoryRegistry, get_default_registry

logger = get_logger(__name__)


@dataclass
class MemCubeConfig:
    """MemCube 配置。"""

    user_id: str = "default"
    agent_id: str | None = None
    enable_metacognition: bool = True
    enable_forgetting: bool = True
    enable_sync: bool = True


class MemCube:
    """MemCube 统一容器 (架构文档 §7.2 自研, 借鉴 MemOS MemCube)。

    封装 Memory facade + 注册表, 提供多类型编排:
    - add: 根据 memory_type 路由到对应 handler
    - search: 元认知路由 + 多类型并行检索
    - analyze: 覆盖报告 + 策略自调

    用法:
        cube = MemCube(Memory(), config=MemCubeConfig(user_id="alice"))
        cube.add("alice likes python", memory_type="semantic")
        results = cube.search("alice likes", memory_type="semantic")
        report = cube.analyze()
    """

    def __init__(
        self, memory: Memory, config: MemCubeConfig | None = None, registry: MemoryRegistry | None = None
    ) -> None:
        self.memory = memory
        self.config = config or MemCubeConfig()
        self.registry = registry or get_default_registry()

    def add(self, content: str, *, memory_type: str = "verbatim", **kwargs: Any) -> dict[str, Any]:
        """根据 memory_type 路由添加 (架构文档 §7.2 编排)。

        memory_type:
        - verbatim: Memory.add (原文存)
        - semantic: Memory.add_fact (三元组)
        - episodic: Memory.add_episode (时序事件)
        - procedural: Memory.add_rule (程序规则)
        """
        uid = kwargs.pop("user_id", self.config.user_id)
        if memory_type == "semantic":
            subject = kwargs.pop("subject", content)
            predicate = kwargs.pop("predicate", "is")
            obj = kwargs.pop("object", "")
            return self.memory.add_fact(subject, predicate, obj, user_id=uid, **kwargs)
        elif memory_type == "episodic":
            return self.memory.add_episode(content, user_id=uid, **kwargs)
        elif memory_type == "procedural":
            return self.memory.add_rule(content, user_id=uid, **kwargs)
        else:
            return self.memory.add(content, user_id=uid, **kwargs)

    def search(
        self, query: str, *, memory_type: str | None = None, top_k: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """根据 memory_type 路由检索 (元认知路由 + 多类型并行)。

        memory_type=None: 元认知路由自动决定查哪些命名空间
        memory_type="verbatim": 基础向量检索
        memory_type="semantic": 语义事实检索
        memory_type="hybrid": BM25+向量 RRF 融合
        memory_type="progressive": 渐进三层检索
        memory_type="strength": 遗忘曲线加权检索
        """
        uid = kwargs.pop("user_id", self.config.user_id)

        if memory_type is None:
            # 元认知路由: L0 自动决定
            route = self.memory.meta_route(query)
            results: list[dict[str, Any]] = []
            for ns in route["namespaces"]:
                results.extend(self._search_namespace(ns, query, user_id=uid, top_k=top_k))
            return results

        return self._search_namespace(memory_type, query, user_id=uid, top_k=top_k)

    def _search_namespace(self, ns: str, query: str, *, user_id: str, top_k: int) -> list[dict[str, Any]]:
        """单命名空间检索。"""
        if ns == "semantic":
            return self.memory.search_facts(query, user_id=user_id, top_k=top_k)
        elif ns == "episodic":
            events = self.memory.get_timeline(user_id=user_id, limit=top_k)
            return [{"id": e["id"], "memory": e.get("content", ""), "score": 0.5} for e in events]
        elif ns == "procedural":
            rules = self.memory.get_active_rules(user_id=user_id)
            return [{"id": r["id"], "memory": r["rule"], "score": r.get("confidence", 0.5)} for r in rules[:top_k]]
        else:
            return self.memory.search(query, user_id=user_id, top_k=top_k)

    def analyze(self) -> dict[str, Any]:
        """元认知分析: 覆盖报告 + 策略自调 (架构文档 §6.3 L1+L2)。"""
        if not self.config.enable_metacognition:
            return {"disabled": True}
        coverage = self.memory.coverage_report(user_id=self.config.user_id)
        strategy = self.memory.adapt_strategy(user_id=self.config.user_id)
        return {"coverage": coverage, "strategy": strategy}

    def rehearse(self) -> dict[str, Any]:
        """主动复述高价值低强度记忆 (架构文档 §6.2)。"""
        if not self.config.enable_forgetting:
            return {"disabled": True}
        candidates = self.memory.find_rehearse_candidates(user_id=self.config.user_id)
        for c in candidates:
            self.memory.rehearse(c["memory_id"], user_id=self.config.user_id)
        return {"rehearsed": len(candidates)}
