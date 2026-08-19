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
"""记忆存储后端抽象基类 + 能力扩展 Protocol。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from septmuse.core.utils import rrf_fuse


class MemoryStore(ABC):
    """记忆存储后端抽象 — 核心 CRUD。"""

    @abstractmethod
    def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆, 返回 memory_id。"""
        ...

    def add_batch(
        self,
        records: list[tuple[str, list[float]]],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> list[str]:
        """批量添加记忆，返回 memory_id 列表。"""
        return [
            self.add(
                content,
                emb,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                metadata=metadata,
                valid_at=valid_at,
            )
            for content, emb in records
        ]

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。"""
        ...

    @abstractmethod
    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """软删除 (标记 is_deleted + history 记录)。"""
        ...

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        ...

    @abstractmethod
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放连接资源。"""
        ...

    # ------------------------------------------------------------------
    # 时态查询默认实现 (不具备时态能力的后端退化为空列表)
    # ------------------------------------------------------------------

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真 — 默认不支持, 返回 NOT_SUPPORTED。"""
        return {"status": "NOT_SUPPORTED", "memory_id": memory_id}

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 — 默认不支持, 返回空列表。"""
        return []

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间内为真的记忆 — 默认不支持, 返回空列表。"""
        return []

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25) — 默认不支持, 返回空列表。"""
        return []

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询访问日志 — 默认不支持, 返回空列表。"""
        return []

    # ------------------------------------------------------------------
    # 混合检索 (默认实现, 依赖 KeywordIndex 能力)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索 (向量 + 关键词 RRF 融合)。"""
        vec_results = self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw_results: list[dict[str, Any]] = []
        if isinstance(self, KeywordIndex):
            kw_results = self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return rrf_fuse(vec_results, kw_results, alpha=alpha)[:top_k]


# ------------------------------------------------------------------
# 能力扩展 Protocol
# ------------------------------------------------------------------


@runtime_checkable
class TemporalStore(Protocol):
    """双时态存储扩展接口。"""

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真。"""
        ...

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。"""
        ...

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。"""
        ...


@runtime_checkable
class KeywordIndex(Protocol):
    """关键词检索扩展接口（BM25）。"""

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。"""
        ...


@runtime_checkable
class AccessLogStore(Protocol):
    """访问日志扩展接口。"""

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志 (审计用)。"""
        ...


@runtime_checkable
class SharedQuery(Protocol):
    """跨 agent 共享查询扩展接口。"""

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        ...

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        ...

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        ...


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
