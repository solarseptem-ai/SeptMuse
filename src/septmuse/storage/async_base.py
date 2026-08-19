"""异步记忆存储后端抽象基类 + 能力扩展 Protocol。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from septmuse.core.utils import rrf_fuse


class AsyncMemoryStore(ABC):
    """异步记忆存储后端抽象 — 核心 CRUD。"""

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
        """向量检索，返回 [{"id", "memory", "score", ...}]。"""
        ...

    @abstractmethod
    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
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
        kw: list[dict[str, Any]] = []
        if isinstance(self, AsyncKeywordIndex):
            kw = await self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return rrf_fuse(vec, kw, alpha=alpha)[:top_k]

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25) — 默认不支持, 返回空列表。"""
        return []

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询访问日志 — 默认不支持, 返回空列表。"""
        return []

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 — 默认不支持, 返回空列表。"""
        return []

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间内为真的记忆 — 默认不支持, 返回空列表。"""
        return []

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真 — 默认不支持。"""
        return {"status": "NOT_SUPPORTED", "memory_id": memory_id}


# ------------------------------------------------------------------
# 能力扩展 Protocol (async 对偶)
# ------------------------------------------------------------------


@runtime_checkable
class AsyncTemporalStore(Protocol):
    """异步双时态存储扩展接口。"""

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]: ...

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class AsyncKeywordIndex(Protocol):
    """异步关键词检索扩展接口（BM25）。"""

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class AsyncAccessLogStore(Protocol):
    """异步访问日志扩展接口。"""

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]: ...
