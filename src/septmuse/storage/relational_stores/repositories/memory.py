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
"""MemoryRepository — memories / history / access_logs 表的纯 SQL CRUD。

接口极窄（add/get/search/delete/update/history），实现深（全部 SQLModel 细节隐藏）。
不触碰 vector_store / keyword_index / engine 生命周期。
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import desc, or_
from sqlalchemy.orm import sessionmaker
from sqlmodel import select

from septmuse.core.logging import get_logger
from septmuse.core.utils import utcnow_iso
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable

logger = get_logger(__name__)

_FILTER_OPERATORS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains",
})


def _apply_operator(stmt: Any, col: Any, op: str, value: Any) -> Any:
    """将单个操作符应用到 SQLAlchemy select 语句。"""
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
    """将 filters 字典应用到 SQLAlchemy select 语句。"""
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


def _mem_to_dict(mem: MemoryTable) -> dict[str, Any]:
    """MemoryTable ORM 对象 → 对外 dict。"""
    return {
        "id": mem.id,
        "memory": mem.content,
        "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
        "created_at": mem.created_at,
        "updated_at": mem.updated_at,
        "state": mem.state or "active",
    }


class MemoryRepository:
    """记忆 SQL CRUD Repository — 接口窄，实现深。

    Args:
        session_maker: SQLAlchemy sessionmaker，由 engine 所有者（DatabaseService）提供。
    """

    def __init__(self, session_maker: sessionmaker) -> None:
        self._session_maker = session_maker

    # ------------------------------------------------------------------
    # 核心 CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        mid: str,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> None:
        """插入一条记忆 + history (ADD)，在 session 内 commit。"""
        now = utcnow_iso()
        with self._session_maker() as session:  # type: ignore[arg-type]
            session.add(MemoryTable(
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

    def add_batch(
        self,
        mids: list[str | None],
        mem_rows: list[MemoryTable],
        hist_rows: list[HistoryTable],
    ) -> None:
        """批量插入记忆 + history，单次 commit。"""
        if not mem_rows:
            return
        with self._session_maker() as session:  # type: ignore[arg-type]
            for mem in mem_rows:
                session.add(mem)
            for h_row in hist_rows:
                session.add(h_row)
            session.commit()

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条 active 记忆，不存在返回 None。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
                or_(MemoryTable.state == "active", MemoryTable.state.is_(None)),
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            return _mem_to_dict(mem)

    def search_by_ids(
        self,
        memory_ids: list[str],
        *,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryTable]:
        """按 ID 列表查询 active 记忆，返回 ORM 对象列表（供 facade 组装 score）。"""
        if not memory_ids:
            return []
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.id.in_(memory_ids),
                MemoryTable.is_deleted == 0,
                or_(MemoryTable.state == "active", MemoryTable.state.is_(None)),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            skip = {"session_id", "run_id"} if session_id is not None else set()
            stmt = _apply_metadata_filters(stmt, filters, MemoryTable, skip_keys=skip)
            return list(session.exec(stmt).all())

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(MemoryTable.state == "active", MemoryTable.state.is_(None)),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            skip = {"session_id", "run_id"} if session_id is not None else set()
            stmt = _apply_metadata_filters(stmt, filters, MemoryTable, skip_keys=skip)
            rows = session.exec(stmt).all()
        return [_mem_to_dict(mem) for mem in rows]

    def delete(self, memory_id: str) -> str | None:
        """软删除，返回旧 content（用于 facade 清理 vector/keyword）。不存在返回 None。"""
        now = utcnow_iso()
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            old_content = mem.content
            mem.is_deleted = 1
            mem.state = "deleted"
            mem.deleted_at = now
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=old_content,
                new_memory=None,
                event="DELETE",
                created_at=now,
                is_deleted=1,
            ))
            session.commit()
            return old_content

    def update(
        self,
        memory_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """更新 content + metadata，返回旧数据 dict（用于 facade 双写更新）。不存在返回 None。"""
        now = utcnow_iso()
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            old_content = mem.content
            old_meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
            result = {
                "old_content": old_content,
                "user_id": mem.user_id,
                "agent_id": mem.agent_id,
                "session_id": mem.session_id,
            }
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
            return result

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
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

    # ------------------------------------------------------------------
    # 时态
    # ------------------------------------------------------------------

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any] | None:
        """标记事实不再为真。不存在返回 None。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            inv = invalid_at or utcnow_iso()
            exp = utcnow_iso()
            now = utcnow_iso()
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
            return {"id": memory_id, "invalid_at": inv, "expired_at": exp, "event": "INVALIDATE"}

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(MemoryTable.valid_at.is_(None), MemoryTable.valid_at <= reference_time),
                or_(MemoryTable.invalid_at.is_(None), MemoryTable.invalid_at > reference_time),
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
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(MemoryTable.valid_at.is_(None), MemoryTable.valid_at <= end),
                or_(MemoryTable.invalid_at.is_(None), MemoryTable.invalid_at > start),
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
    # 访问日志
    # ------------------------------------------------------------------

    def record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """记录记忆访问日志。"""
        log_id = str(uuid.uuid4())
        now = utcnow_iso()
        with self._session_maker() as session:  # type: ignore[arg-type]
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
        """查询记忆访问日志。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
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
    # 关系查询 (跨 agent 共享)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable.agent_id).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                MemoryTable.agent_id.isnot(None),
            ).distinct()
            return list(session.exec(stmt).all())

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable.user_id).where(
                MemoryTable.agent_id == agent_id,
                MemoryTable.is_deleted == 0,
            ).distinct()
            return list(session.exec(stmt).all())

    def get_user_ids(self) -> list[str]:
        """列出所有有活跃记忆的 user_id (去重)。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable.user_id).where(
                MemoryTable.state == "active",
            ).distinct()
            return [uid for uid in session.exec(stmt).all() if uid]

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
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

    # ------------------------------------------------------------------
    # 关键词检索辅助
    # ------------------------------------------------------------------

    def get_by_id_for_keyword(
        self, doc_id: str, *, user_id: str, session_id: str | None = None
    ) -> dict[str, Any] | None:
        """关键词检索后按 ID 查内容。"""
        with self._session_maker() as session:  # type: ignore[arg-type]
            stmt = select(MemoryTable).where(
                MemoryTable.id == doc_id,
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            return _mem_to_dict(mem)
