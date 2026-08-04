"""异步记忆存储后端抽象基类。

所有方法为 async def，使用 aiosqlite/asyncpg 等异步驱动。
sync MemoryStore 的对偶，方法签名保持一致。
score 语义: 相似度 (越高越相似, 范围 [0, 1])。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AsyncMemoryStore(ABC):
    """异步记忆存储后端抽象。

    实现方需保证:
    - add 返回唯一 memory_id
    - search 的 score 为相似度 (0-1, 越高越相似)
    - delete 为软删除 (标记 is_deleted + history 记录)
    - user_id 隔离 (不同用户互不可见)
    """

    @abstractmethod
    async def add(
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
        """添加记忆，返回 memory_id。"""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{"id", "memory", "score", ...}]。

        filters: mem0 风格字段过滤字典, None=不过滤 (子类按需实现)。
        """
        ...

    @abstractmethod
    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。

        filters: mem0 风格字段过滤字典, None=不过滤 (子类按需实现)。
        """
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条。"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """软删除。"""
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆。"""
        ...

    @abstractmethod
    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取变更历史。"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """释放连接资源。"""
        ...

    # ── 默认实现（子类可覆盖）──

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索。默认返回空。"""
        return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索（向量 + 关键词 RRF 融合）。"""
        vec = await self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw = await self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return _rrf_fuse(vec, kw, alpha=alpha)[:top_k]

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询访问日志。默认返回空。"""
        return []

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。默认返回空。"""
        return []

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间内为真的记忆。默认返回空。"""
        return []

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真。默认不支持。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 invalidate")


def _rrf_fuse(vec_results: list[dict], kw_results: list[dict], *, alpha: float = 0.5, k: int = 60) -> list[dict]:
    """RRF 融合排序（纯计算，无 I/O，不需 async）。

    score = alpha * 1/(k+rank_vec) + (1-alpha) * 1/(k+rank_kw)
    alpha=1.0 时仅保留向量侧; alpha=0.0 时仅保留关键词侧。
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
