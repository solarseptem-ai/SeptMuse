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
"""ORMMemoryStore — SQLModel ORM 跨方言记忆存储。

一套代码跑 SQLite/MySQL/PostgreSQL。从 DatabaseService 拿 engine，
CRUD 全用 SQLModel select() / session.add()。SQLModel.metadata.create_all()
自动生成对应方言的 DDL。

向量以 JSON list[float] 存储, 检索用 numpy 余弦相似 (跨方言通用)。
组合 vector_store + keyword_index 双写 (方言工厂创建, P3/P4 实现)。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, or_
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel, select

from septmuse.core.logging import get_logger
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 支持的过滤操作符
_FILTER_OPERATORS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains",
})


def _apply_operator(stmt: Any, col: Any, op: str, value: Any) -> Any:
    """将单个操作符应用到 SQLAlchemy select 语句 (对齐 mem0 高级过滤)。"""
    if op == "eq":
        return stmt.where(col == value)
    if op == "ne":
        return stmt.where(col != value)
    if op == "gt":
        return stmt.where(col > value)
    if op == "gte":
        return stmt.where(col >= value)
    if op == "lt":
        return stmt.where(col < value)
    if op == "lte":
        return stmt.where(col <= value)
    if op == "in":
        return stmt.where(col.in_(value))
    if op == "nin":
        return stmt.where(~col.in_(value))
    if op == "contains":
        return stmt.where(col.contains(value))
    if op == "icontains":
        return stmt.where(col.icontains(value))
    return stmt


def _apply_metadata_filters(
    stmt: Any,
    filters: dict[str, Any] | None,
    model: type,
    *,
    skip_keys: set[str] | None = None,
) -> Any:
    """将 filters 字典应用到 SQLAlchemy select 语句。

    支持两种格式:
    - 精确匹配: {"key": "value"} (向后兼容)
    - 操作符: {"key": {"eq"/"ne"/"gt"/"gte"/"lt"/"lte"/"in"/"nin"/"contains"/"icontains": value}}
    """
    if not filters:
        return stmt
    clean = dict(filters)
    for skip in (skip_keys or set()):
        clean.pop(skip, None)
    for key, value in clean.items():
        if not hasattr(model, key):
            continue
        col = getattr(model, key)
        if isinstance(value, dict):
            for op, op_val in value.items():
                if op in _FILTER_OPERATORS:
                    stmt = _apply_operator(stmt, col, op, op_val)
        else:
            stmt = stmt.where(col == value)
    return stmt


class ORMMemoryStore(MemoryStore):
    """SQLModel ORM 记忆存储 — 跨方言 CRUD。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        store = ORMMemoryStore(engine)
        mid = store.add("hello", [0.1, 0.2], user_id="alice")
    """

    def __init__(
        self,
        engine: Engine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
    ) -> None:
        self._engine = engine
        self._session_maker = sessionmaker(engine, expire_on_commit=False)
        if vector_store is None:
            from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

            vector_store = SQLAlchemyVectorStore(engine)
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._create_tables()
        logger.info("orm_store_ready", dialect=engine.dialect.name)

    @property
    def engine(self) -> Engine:
        """暴露内部 engine，供 facade duck typing 取用。"""
        return self._engine

    def _create_tables(self) -> None:
        """建表 — SQLModel.metadata.create_all 跨方言 DDL + MigrationRunner 补旧 DB 缺列。"""
        SQLModel.metadata.create_all(self._engine)
        # 运行迁移: 旧 DB (SQLiteMemoryStore 创建) 可能缺 archived_at 等列
        from septmuse.storage.migrations.runner import MigrationRunner

        MigrationRunner(self._engine).run()

    def close(self) -> None:
        """释放引擎资源。"""
        self._engine.dispose()

    def reset(self) -> None:
        """重置存储 (删表数据 + 双写清理).

        清除 memories / history / memory_access_logs 三表数据,
        并重置 vector_store + keyword_index。
        """
        from sqlalchemy import text

        with Session(self._engine) as session:
            for table_name in ["memories", "history", "memory_access_logs"]:
                with contextlib.suppress(Exception):
                    session.exec(text(f"DELETE FROM {table_name}"))
            session.commit()
        # 双写清理: vector_store
        if self._vector_store is not None:
            try:
                if hasattr(self._vector_store, "reset_collection"):
                    self._vector_store.reset_collection()
                elif hasattr(self._vector_store, "delete_collection"):
                    self._vector_store.delete_collection()
            except Exception:
                pass
        # 双写清理: keyword_index
        if self._keyword_index is not None:
            try:
                if hasattr(self._keyword_index, "reset"):
                    self._keyword_index.reset()
            except Exception:
                pass
        logger.info("orm_store_reset_done")

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
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        with Session(self._engine) as session:
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
            session.commit()
        # 双写: vector_store + keyword_index
        if self._vector_store is not None:
            self._vector_store.insert_vectors(
                [embedding], [mid], [{"user_id": user_id, "session_id": session_id}]
            )
        if self._keyword_index is not None:
            self._keyword_index.add_docs({mid: content})
        logger.info("memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

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
        """批量添加记忆, 返回 memory_id 列表 (单次 commit, 对齐 mem0 V3 Phase 6)。

        含批次内 hash 去重 (MD5, 对齐 mem0 Phase 5): 同批次相同内容的记录只存一条。
        跨批次去重靠 P0-1 增量决策 (已有记忆注入 prompt)。

        Args:
            records: [(content, embedding), ...] 列表
            user_id: 用户 ID (必填)
            agent_id / session_id / metadata / valid_at: 同 add()

        Returns:
            memory_id 列表, 顺序与 records 一致 (去重的记录 id 为 None)
        """
        if not records:
            return []
        now = _utcnow_iso()
        mids: list[str | None] = [None] * len(records)
        mem_rows: list[MemoryTable] = []
        hist_rows: list[HistoryTable] = []
        all_embeddings: list[list[float]] = []
        all_ids: list[str] = []
        all_docs: dict[str, str] = {}
        seen_hashes: set[str] = set()

        for idx, (content, embedding) in enumerate(records):
            h = hashlib.md5(content.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            mid = f"mem-{uuid.uuid4()}"
            mids[idx] = mid
            mem_rows.append(MemoryTable(
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
            ))
            hist_rows.append(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=mid,
                old_memory=None,
                new_memory=content,
                event="ADD",
                created_at=now,
                is_deleted=0,
            ))
            all_embeddings.append(embedding)
            all_ids.append(mid)
            all_docs[mid] = content

        if not mem_rows:
            logger.info("memory_batch_all_duplicates", count=len(records), user_id=user_id)
            return mids

        with Session(self._engine) as session:
            for mem in mem_rows:
                session.add(mem)
            for h_row in hist_rows:
                session.add(h_row)
            session.commit()

        # 批量双写
        if self._vector_store is not None and all_embeddings:
            self._vector_store.insert_vectors(
                all_embeddings, all_ids,
                [{"user_id": user_id, "session_id": session_id}] * len(all_ids),
            )
        if self._keyword_index is not None and all_docs:
            self._keyword_index.add_docs(all_docs)
        logger.info(
            "memory_batch_added", count=len(all_ids), duplicates=len(records) - len(all_ids), user_id=user_id
        )
        return mids

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            return {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "state": mem.state or "active",
            }

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
        """向量检索 (委托给 _vector_store ANN 索引, 不再自行 numpy 余弦)。

        流程: _vector_store.search_vectors → 取 ID → MemoryTable 查 content + metadata + 高级过滤。
        score: 相似度 (越高越相似, 范围 [0, 1])。
        """
        if self._vector_store is None:
            return []

        # session_id 可来自参数或 filters 字典
        effective_session_id = session_id
        if effective_session_id is None and filters:
            effective_session_id = filters.get("session_id")

        vs_filters: dict[str, Any] = {"user_id": user_id}
        if effective_session_id is not None:
            vs_filters["session_id"] = effective_session_id

        vec_results = self._vector_store.search_vectors(
            query_embedding, top_k=top_k * 5, filters=vs_filters
        )
        if not vec_results:
            return []

        vec_results = [r for r in vec_results if r.score >= threshold]
        if not vec_results:
            return []

        score_map = {r.id: r.score for r in vec_results}

        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id.in_(list(score_map.keys())),
                MemoryTable.is_deleted == 0,
            )
            if effective_session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == effective_session_id)
            stmt = _apply_metadata_filters(stmt, filters, MemoryTable, skip_keys={"session_id", "run_id"})
            rows = session.exec(stmt).all()

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

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            # 高级过滤 (eq/ne/gt/gte/lt/lte/in/nin/contains/icontains, 对齐 mem0)
            skip = {"session_id", "run_id"} if session_id is not None else set()
            stmt = _apply_metadata_filters(stmt, filters, MemoryTable, skip_keys=skip)
            rows = session.exec(stmt).all()
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

    def delete(self, memory_id: str) -> None:
        """软删除 (标记 is_deleted + state='deleted' + history 记录)。"""
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
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
            session.commit()
        # 双写清理
        if self._vector_store is not None:
            self._vector_store.delete_vector(memory_id)
        if self._keyword_index is not None:
            self._keyword_index.delete_docs([memory_id])
        logger.info("memory_deleted", memory_id=memory_id)

    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return False
            old_content = mem.content
            old_meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
            vs_user_id = mem.user_id
            vs_session_id = mem.session_id
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
            session.commit()
        # 双写更新 (带 payload, 否则过滤搜索找不到更新后的向量)
        if self._vector_store is not None:
            self._vector_store.insert_vectors(
                [embedding], [memory_id],
                [{"user_id": vs_user_id, "session_id": vs_session_id}],
            )
        if self._keyword_index is not None:
            self._keyword_index.add_docs({memory_id: content})
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        with Session(self._engine) as session:
            stmt = select(HistoryTable).where(
                HistoryTable.memory_id == memory_id
            ).order_by(HistoryTable.created_at)
            rows = session.exec(stmt).all()
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

    def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志（审计用）。"""
        log_id = str(uuid.uuid4())
        now = _utcnow_iso()
        with Session(self._engine) as session:
            session.add(AccessLogTable(
                id=log_id,
                memory_id=memory_id,
                app_id=app_id,
                access_type=access_type,
                metadata_json=json.dumps(metadata) if metadata else None,
                accessed_at=now,
            ))
            session.commit()
        return log_id

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志（按 accessed_at 降序）。"""
        with Session(self._engine) as session:
            stmt = select(AccessLogTable).where(
                AccessLogTable.memory_id == memory_id
            ).order_by(AccessLogTable.accessed_at.desc()).limit(limit)
            rows = session.exec(stmt).all()
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

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真（设置 invalid_at + expired_at, 不删除记忆）。"""
        existing = self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        inv = invalid_at or _utcnow_iso()
        exp = _utcnow_iso()
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
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
            session.commit()
        logger.info("memory_invalidated", memory_id=memory_id, invalid_at=inv)
        return {"id": memory_id, "invalid_at": inv, "expired_at": exp, "event": "INVALIDATE"}

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 (valid_at <= t AND (invalid_at IS NULL OR invalid_at > t))。"""
        with Session(self._engine) as session:
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
            rows = session.exec(stmt).all()
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

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。"""
        with Session(self._engine) as session:
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
            rows = session.exec(stmt).all()
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

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。无 keyword_index 时返回空。"""
        if self._keyword_index is None:
            return []
        scores = self._keyword_index.retrieve(query, top_k * 2)
        if not scores:
            return []
        results: list[dict[str, Any]] = []
        with Session(self._engine) as session:
            for doc_id, score in scores.items():
                stmt = select(MemoryTable).where(
                    MemoryTable.id == doc_id,
                    MemoryTable.user_id == user_id,
                    MemoryTable.is_deleted == 0,
                )
                if session_id is not None:
                    stmt = stmt.where(MemoryTable.session_id == session_id)
                mem = session.exec(stmt).first()
                if mem is not None:
                    results.append({
                        "id": mem.id,
                        "memory": mem.content,
                        "score": float(score),
                        "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                        "created_at": mem.created_at,
                    })
        return results[:top_k]

    # ------------------------------------------------------------------
    # 关系查询 (跨 agent 共享)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable.agent_id).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                MemoryTable.agent_id.isnot(None),
            ).distinct()
            return list(session.exec(stmt).all())

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable.user_id).where(
                MemoryTable.agent_id == agent_id,
                MemoryTable.is_deleted == 0,
            ).distinct()
            return list(session.exec(stmt).all())

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            ).order_by(desc(MemoryTable.created_at)).limit(limit)
            rows = session.exec(stmt).all()
        return [
            {
                "id": mem.id,
                "user_id": mem.user_id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "agent_id": mem.agent_id,
            }
            for mem in rows
        ]
