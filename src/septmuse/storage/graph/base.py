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
"""图存储后端抽象基类 (借鉴 cognee GraphDBInterface 简化为同步 + SeptMuse 所需方法)。

所有图后端 (SQLiteGraphStore / AGEGraphStore / 未来 Neo4jGraphStore 等)
实现此接口, 保证 zettel/dream 等演化模块可插拔。

图语义: 记忆间有向边 (source_id → target_id), 每条边有 relation + score。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GraphEdge:
    """图边 (对齐 zettel.MemoryLink, 重命名为通用图语义)。"""

    id: str
    source_id: str
    target_id: str
    relation: str
    score: float


class GraphStore(ABC):
    """图存储后端抽象。

    实现方需保证:
    - add_edge 幂等 (重复 source+target+relation 不报错, 用 UNIQUE 约束或 MERGE)
    - get_edges 返回出边 (source_id 为给定节点的边)
    - get_neighbors 可按 relation 过滤
    - 双向链接由调用方分两次 add_edge (不内置, 保持图的数学语义)
    """

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
