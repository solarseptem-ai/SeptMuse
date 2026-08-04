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
"""记忆存储后端抽象基类。

所有存储后端 (ORMMemoryStore / PGVectorStore / 未来 Qdrant 等)
实现此接口, 保证 capture/retrieval/evolution 等横切关注点可插拔。

方法签名严格对齐 ORMMemoryStore 既有实现, 不破坏现有行为。
score 语义: 相似度 (越高越相似, 范围 [0, 1])。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """记忆存储后端抽象。

    实现方需保证:
    - add 返回唯一 memory_id
    - search 的 score 为相似度 (0-1, 越高越相似)
    - delete 为软删除 (标记 is_deleted + history 记录)
    - user_id 隔离 (不同用户互不可见)
    """

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
        """添加记忆, 返回 memory_id。

        session_id: 会话 ID (可选, 用于会话级过滤; None=不限制)。
        """
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
        """批量添加记忆, 返回 memory_id 列表。

        默认实现: 逐条调用 add()。子类可覆盖以提高性能 (单次 commit)。
        """
        return [
            self.add(
                content, emb,
                user_id=user_id, agent_id=agent_id,
                session_id=session_id, metadata=metadata, valid_at=valid_at,
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
        """向量检索, 返回 [{"id", "memory", "score", "metadata", "created_at"}]。

        session_id: 仅搜该会话的记忆 (None=不限)。
        score: 相似度 (越高越相似, 范围 [0, 1])。
        threshold: 最低相似度过滤阈值。
        filters: 字段过滤字典, None=不过滤 (子类按需实现)。
        """
        ...

    @abstractmethod
    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。

        session_id: 仅返回该会话的记忆 (None=不限)。
        filters: 字段过滤字典, None=不过滤 (子类按需实现)。
        """
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
        """更新记忆 content + embedding + metadata, 记录 history。

        Returns:
            True = 更新成功; False = memory_id 不存在或已删除
        """
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
    # 关系查询 (跨 agent 共享)
    # ------------------------------------------------------------------

    @abstractmethod
    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        ...

    @abstractmethod
    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        ...

    @abstractmethod
    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        ...

    # ------------------------------------------------------------------
    # 混合检索 (向量 + 关键词 RRF 融合, 子类有 KeywordIndex 时覆盖 keyword_search)
    # ------------------------------------------------------------------

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。默认返回空 (子类有 KeywordIndex 时覆盖)。

        session_id: 仅搜该会话的记忆 (None=不限)。
        返回格式同 search: [{"id", "memory", "score", "metadata", "created_at"}]
        """
        return []

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
        """混合检索 (向量 + 关键词 RRF 融合)。

        session_id: 仅搜该会话的记忆 (None=不限)。
        alpha: 向量权重 [0,1]。0=纯关键词, 1=纯向量, 0.5=均衡。
        默认实现: 向量 search + 关键词 keyword_search, RRF 融合排序。
        """
        vec_results = self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw_results = self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return _rrf_fuse(vec_results, kw_results, alpha=alpha)[:top_k]

    # ------------------------------------------------------------------
    # 访问日志 (P2 权限层, 子类有日志表时覆盖)
    # ------------------------------------------------------------------

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志 (审计用)。默认返回空 (子类有日志表时覆盖)。"""
        return []

    # ------------------------------------------------------------------
    # 时态 (双时态建模, 子类有 temporal 列时覆盖)
    # ------------------------------------------------------------------

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 (valid_at <= t AND (invalid_at IS NULL OR invalid_at > t))。

        session_id: 仅返回该会话的记忆 (None=不限)。
        默认返回空 (子类有 temporal 列时覆盖)。
        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
        """
        return []

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。

        条件: valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)
        session_id: 仅返回该会话的记忆 (None=不限)。
        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。

        默认返回空 (子类有 temporal 列时覆盖)。
        """
        return []

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。

        默认不支持 (子类有 temporal 列时覆盖)。
        """
        raise NotImplementedError(f"{type(self).__name__} 不支持 invalidate (需 temporal 列)")


def _rrf_fuse(vec_results: list[dict], kw_results: list[dict], *, alpha: float = 0.5, k: int = 60) -> list[dict]:
    """RRF 融合排序 (k=60 标准参数)。

    score = alpha * 1/(k+rank_vec) + (1-alpha) * 1/(k+rank_kw)

    alpha=1.0 时仅保留向量侧; alpha=0.0 时仅保留关键词侧 (纯模式不混入零分项)。
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if alpha > 0:
        for rank, r in enumerate(vec_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + alpha / (k + rank + 1)
            meta.setdefault(r["id"], r)
    if alpha < 1:
        for rank, r in enumerate(kw_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + (1 - alpha) / (k + rank + 1)
            meta.setdefault(r["id"], r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[mid], "score": sc} for mid, sc in ordered]
