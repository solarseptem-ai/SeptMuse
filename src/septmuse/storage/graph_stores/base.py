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
"""图存储后端抽象基类 — 节点 + 有向边，供演化模块可插拔使用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEdge:
    """图边 (对齐 zettel.MemoryLink, 重命名为通用图语义)。"""

    id: str
    source_id: str
    target_id: str
    relation: str
    score: float
    properties: dict[str, Any] = field(default_factory=dict)


class GraphStore(ABC):
    """图存储后端抽象。"""

    # ── 边管理 (原有, 抽象) ──

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        score: float = 0.0,
    ) -> str:
        """添加有向边, 返回 edge_id。幂等 (重复不报错)。"""
        ...

    @abstractmethod
    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """获取节点的所有出边 (source_id == node_id)。"""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
    ) -> list[str]:
        """获取邻居节点 ID 列表。relation=None 表示所有关系。"""
        ...

    @abstractmethod
    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        """检查边是否存在。"""
        ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """删除边。True=删除成功, False=不存在。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        ...

    # ── 节点管理 (新增, 抽象) ──

    @abstractmethod
    def add_node(self, node_id: str, properties: dict[str, Any] | None = None) -> None:
        """添加或更新节点 (幂等, 存在则更新 properties)。"""
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """获取节点属性。None=不存在。"""
        ...

    @abstractmethod
    def delete_node(self, node_id: str, *, cascade: bool = True) -> bool:
        """删除节点。cascade=True 同时删除关联边。返回是否删除成功。"""
        ...

    # ── 入边查询 (新增, 抽象) ──

    @abstractmethod
    def get_in_edges(self, node_id: str) -> list[GraphEdge]:
        """获取节点的所有入边 (target_id == node_id)。"""
        ...

    # ── 图统计 (新增, 抽象) ──

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """图统计: node_count, edge_count, density (2*edges/nodes*(nodes-1))。"""
        ...

    # ── 多跳遍历 (新增, 非抽象, 默认 Python BFS fallback) ──

    def traverse(
        self,
        seed_id: str,
        max_depth: int = 2,
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """BFS 遍历，返回 [{"id", "depth"}] (不含种子节点)。"""
        if max_depth < 1:
            return []

        visited: set[str] = {seed_id}
        results: list[dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque([(seed_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            if direction == "in":
                neighbors = [e.source_id for e in self.get_in_edges(current_id)]
            elif direction == "both":
                out_n = self.get_neighbors(current_id, relation)
                in_n = [e.source_id for e in self.get_in_edges(current_id)]
                neighbors = list(dict.fromkeys(out_n + in_n))
            else:
                neighbors = self.get_neighbors(current_id, relation)

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                results.append({"id": neighbor_id, "depth": depth + 1})
                queue.append((neighbor_id, depth + 1))

        return results

    # ── 社区检测 (新增, 非抽象, 默认 NotImplementedError) ──

    def detect_communities(self, algorithm: str = "label_propagation") -> dict[str, list[str]]:
        """社区检测，返回 {community_id: [node_ids]}。"""
        raise NotImplementedError(f"Community detection '{algorithm}' not implemented for {type(self).__name__}")
