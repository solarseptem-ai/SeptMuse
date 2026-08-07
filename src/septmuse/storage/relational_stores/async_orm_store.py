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
"""AsyncORMMemoryStore — SQLModel async ORM 跨方言记忆存储。

ORMMemoryStore 的 async 对偶。用 AsyncSession + async_sessionmaker。
双写 vector_store/keyword_index 用 asyncio.to_thread 包装 sync 调用。
建表用 sync engine (从 async URL 派生), 避免 event loop 嵌套问题。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, or_
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from septmuse.core.logging import get_logger
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable
from septmuse.storage.async_base import AsyncMemoryStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AsyncORMMemoryStore(AsyncMemoryStore):
    """SQLModel async ORM 记忆存储 — 跨方言 CRUD。

    用法:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine("sqlite+aiosqlite:///test.db")
        store = AsyncORMMemoryStore(engine)
        mid = await store.add("hello", [0.1, 0.2], user_id="alice")
    """

    def __init__(
        self,
        engine: AsyncEngine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
    ) -> None:
        self._engine = engine
        self._session_maker = async_sessionmaker(engine, expire_on_commit=False)
        self._keyword_index = keyword_index
        # 建表: 从 async URL 派生 sync engine 建表 (避免 event loop 嵌套问题)
        # in-memory SQLite 因连接隔离不适用此方式, 测试请用文件 DB
        sync_url = (
            str(engine.url)
            .replace("+aiosqlite", "")
            .replace("+aiomysql", "+pymysql")
            .replace("+asyncpg", "+psycopg2")
        )
        sync_engine = create_engine(sync_url)
        try:
            SQLModel.metadata.create_all(sync_engine)
        except Exception:
            sync_engine.dispose()
            raise
        if vector_store is None:
            from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

            vector_store = SQLAlchemyVectorStore(sync_engine)
        else:
            sync_engine.dispose()
        self._vector_store = vector_store
        logger.info("async_orm_store_ready", dialect=engine.dialect.name)

    @property
    def async_engine(self) -> AsyncEngine:
        """暴露内部 async engine，供 async facade duck typing 取用。"""
        return self._engine

    async def close(self) -> None:
        """释放引擎资源。"""
        await self._engine.dispose()

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
        """添加记忆, 返回 memory_id。"""
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            mem = MemoryTable(
                id=mid,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                metadata_json=json.dumps(metadata or {}),
                created_at=now,
                updated_at=now,
                valid_at=valid_at,
                is_deleted=0,
                state="active",
            )
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=mid,
                old_memory=None,
                new_memory=content,
                event="ADD",
                created_at=now,
                is_deleted=0,
            ))
            await session.commit()
        # 双写: vector_store + keyword_index (sync, 用 to_thread)
        if self._vector_store is not None:
            await asyncio.to_thread(
                self._vector_store.insert_vectors,
                [embedding],
                [mid],
                [{"user_id": user_id, "session_id": session_id}],
            )
        if self._keyword_index is not None:
            await asyncio.to_thread(self._keyword_index.add_docs, {mid: content})
        logger.info("async_memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            result = await session.exec(stmt)
            mem = result.first()
            if mem is None:
                return None
            return {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "state": mem.state or "active",
            }

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
        """向量检索 (委托给 _vector_store ANN 索引)。

        score: 相似度 (越高越相似, 范围 [0, 1])。
        """
        if self._vector_store is None:
            return []

        vs_filters: dict[str, Any] = {"user_id": user_id}
        if session_id is not None:
            vs_filters["session_id"] = session_id

        vec_results = await asyncio.to_thread(
            self._vector_store.search_vectors,
            query_embedding, top_k * 3, vs_filters,
        )
        if not vec_results:
            return []

        vec_results = [r for r in vec_results if r.score >= threshold]
        if not vec_results:
            return []

        score_map = {r.id: r.score for r in vec_results}

        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id.in_(list(score_map.keys())),
                MemoryTable.is_deleted == 0,
            )
            if filters:
                clean_filters = dict(filters)
                clean_filters.pop("session_id", None)
                clean_filters.pop("run_id", None)
                for key, value in clean_filters.items():
                    if hasattr(MemoryTable, key):
                        stmt = stmt.where(getattr(MemoryTable, key) == value)
            result = await session.exec(stmt)
            rows = result.all()

        results: list[dict[str, Any]] = []
        for mem in rows:
            results.append({
                "id": mem.id,
                "memory": mem.content,
                "score": score_map.get(mem.id, 0.0),
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            if filters:
                clean_filters = dict(filters)
                if session_id is not None:
                    clean_filters.pop("session_id", None)
                    clean_filters.pop("run_id", None)
                for key, value in clean_filters.items():
                    if hasattr(MemoryTable, key):
                        stmt = stmt.where(getattr(MemoryTable, key) == value)
            result = await session.exec(stmt)
            rows = result.all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "updated_at": mem.updated_at,
            }
            for mem in rows
        ]

    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            result = await session.exec(stmt)
            mem = result.first()
            if mem is None:
                return False
            old_content = mem.content
            old_meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
            mem.content = content
            mem.metadata_json = json.dumps(metadata if metadata is not None else old_meta)
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=old_content,
                new_memory=content,
                event="UPDATE",
                created_at=now,
                is_deleted=0,
            ))
            await session.commit()
        # 双写更新
        if self._vector_store is not None:
            await asyncio.to_thread(self._vector_store.insert_vectors, [embedding], [memory_id])
        if self._keyword_index is not None:
            await asyncio.to_thread(self._keyword_index.add_docs, {memory_id: content})
        logger.info("async_memory_updated", memory_id=memory_id, content_len=len(content))
        return True

    async def delete(self, memory_id: str) -> None:
        """软删除 (标记 is_deleted + state='deleted' + history 记录)。"""
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            result = await session.exec(stmt)
            mem = result.first()
            if mem is None:
                return
            mem.is_deleted = 1
            mem.state = "deleted"
            mem.deleted_at = now
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=mem.content,
                new_memory=None,
                event="DELETE",
                created_at=now,
                is_deleted=1,
            ))
            await session.commit()
        # 双写清理 (sync, 用 to_thread)
        if self._vector_store is not None:
            await asyncio.to_thread(self._vector_store.delete_vector, memory_id)
        if self._keyword_index is not None:
            await asyncio.to_thread(self._keyword_index.delete_docs, [memory_id])
        logger.info("async_memory_deleted", memory_id=memory_id)

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(HistoryTable).where(
                HistoryTable.memory_id == memory_id
            ).order_by(HistoryTable.created_at)
            result = await session.exec(stmt)
            rows = result.all()
        return [
            {
                "id": h.id,
                "memory_id": h.memory_id,
                "old_memory": h.old_memory,
                "new_memory": h.new_memory,
                "event": h.event,
                "created_at": h.created_at,
                "is_deleted": bool(h.is_deleted),
            }
            for h in rows
        ]

    async def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志（审计用）。"""
        log_id = str(uuid.uuid4())
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            session.add(AccessLogTable(
                id=log_id,
                memory_id=memory_id,
                app_id=app_id,
                access_type=access_type,
                metadata_json=json.dumps(metadata) if metadata else None,
                accessed_at=now,
            ))
            await session.commit()
        return log_id

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志（按 accessed_at 降序）。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(AccessLogTable).where(
                AccessLogTable.memory_id == memory_id
            ).order_by(AccessLogTable.accessed_at.desc()).limit(limit)
            result = await session.exec(stmt)
            rows = result.all()
        return [
            {
                "id": log.id,
                "memory_id": log.memory_id,
                "app_id": log.app_id,
                "access_type": log.access_type,
                "metadata": json.loads(log.metadata_json) if log.metadata_json else None,
                "accessed_at": log.accessed_at,
            }
            for log in rows
        ]

    # ------------------------------------------------------------------
    # 时态 (双时态建模, 借鉴 graphiti EntityEdge)
    # ------------------------------------------------------------------

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真（设置 invalid_at + expired_at, 不删除记忆）。"""
        existing = await self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        inv = invalid_at or _utcnow_iso()
        exp = _utcnow_iso()
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            result = await session.exec(stmt)
            mem = result.first()
            if mem is None:
                return {"id": memory_id, "event": "NOT_FOUND"}
            mem.invalid_at = inv
            mem.expired_at = exp
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=mem.content,
                new_memory=None,
                event="INVALIDATE",
                created_at=now,
                is_deleted=0,
            ))
            await session.commit()
        logger.info("async_memory_invalidated", memory_id=memory_id, invalid_at=inv)
        return {"id": memory_id, "invalid_at": inv, "expired_at": exp, "event": "INVALIDATE"}

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 (valid_at <= t AND (invalid_at IS NULL OR invalid_at > t))。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(
                    MemoryTable.valid_at.is_(None),  # 无时间约束, 始终返回
                    MemoryTable.valid_at <= reference_time,
                ),
                or_(
                    MemoryTable.invalid_at.is_(None),  # 未失效
                    MemoryTable.invalid_at > reference_time,
                ),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            result = await session.exec(stmt)
            rows = result.all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "valid_at": mem.valid_at,
                "invalid_at": mem.invalid_at,
            }
            for mem in rows
        ]

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(
                    MemoryTable.valid_at.is_(None),
                    MemoryTable.valid_at <= end,
                ),
                or_(
                    MemoryTable.invalid_at.is_(None),
                    MemoryTable.invalid_at > start,
                ),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            result = await session.exec(stmt)
            rows = result.all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "valid_at": mem.valid_at,
                "invalid_at": mem.invalid_at,
            }
            for mem in rows
        ]

    # ------------------------------------------------------------------
    # 关键词检索 (BM25, 依赖 keyword_index)
    # ------------------------------------------------------------------

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。无 keyword_index 时返回空。"""
        if self._keyword_index is None:
            return []
        scores = await asyncio.to_thread(self._keyword_index.retrieve, query, top_k * 2)
        if not scores:
            return []
        results: list[dict[str, Any]] = []
        async with AsyncSession(self._engine) as session:
            for doc_id, score in scores.items():
                stmt = select(MemoryTable).where(
                    MemoryTable.id == doc_id,
                    MemoryTable.user_id == user_id,
                    MemoryTable.is_deleted == 0,
                )
                if session_id is not None:
                    stmt = stmt.where(MemoryTable.session_id == session_id)
                result = await session.exec(stmt)
                mem = result.first()
                if mem is not None:
                    results.append({
                        "id": mem.id,
                        "memory": mem.content,
                        "score": float(score),
                        "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                        "created_at": mem.created_at,
                    })
        return results[:top_k]
