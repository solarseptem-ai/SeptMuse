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
"""SQLite 零配置组合后端 — SeptMuse 默认存储 (架构文档 §12.2, §3 storage/sqlite/)。

组合 memories + history 于单 SQLite 文件, 向量检索回退 numpy 余弦
(sqlite-vec 不可用时, 架构文档 §12.8 设计)。

参考模式 (实证, 非自行发挥):
- 事务/表结构: mem0/mem0/memory/storage.py SQLiteManager (sqlite3 + threading.Lock + BEGIN/COMMIT/ROLLBACK)
- add/search/get_all/delete 契约: mem0 Memory 方法签名
- history 表字段: mem0 SQLiteManager history 表 (memory_id/old_memory/new_memory/event/created_at/is_deleted)

P0-P3 扩展:
- 组合 vector_store + keyword_index 双写 (test_composite_store)
- state 状态机 (active/paused/archived/deleted) + ALTER TABLE 迁移
- session_id 会话隔离 (对齐 mem0 run_id)
- 双时态 valid_at/invalid_at/expired_at
- memory_access_logs 审计日志
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore, _rrf_fuse
from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index
from septmuse.storage.vector.sqlite_vec import SQLiteVectorStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_db_path() -> Path:
    """默认 db 路径: ~/.septmuse/septmuse.db (零配置)。"""
    home = Path.home()
    return home / ".septmuse" / "septmuse.db"


class SQLiteMemoryStore(MemoryStore):
    """零配置组合后端: memories + history 单文件。

    向量以 JSON list[float] 存储, 检索用 numpy 余弦相似 (sqlite-vec 优化后续)。
    组合 _vector_store (SQLiteVectorStore) + _keyword_index (SQLiteBM25Index) 双写。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _default_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()
        # 轻量级迁移：版本跟踪 + 有序迁移（替代 _migrate_add_* 方法）
        from septmuse.storage.migrations.runner import MigrationRunner
        MigrationRunner(self.conn, "sqlite").run()
        # 组合后端: vector_store 共享 conn, keyword_index 共享 db_path
        self._vector_store = SQLiteVectorStore(conn=self.conn, lock=self._lock)
        bm25_path = str(self.db_path)
        self._keyword_index = SQLiteBM25Index(db_path=bm25_path)
        logger.info("sqlite_store_ready", path=str(self.db_path))

    def _create_tables(self) -> None:
        """建表 (参考 mem0 SQLiteManager _create_*_table 模式)。"""
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id          TEXT PRIMARY KEY,
                        user_id     TEXT NOT NULL,
                        agent_id    TEXT,
                        content     TEXT NOT NULL,
                        embedding   TEXT NOT NULL,
                        metadata    TEXT,
                        created_at  TEXT,
                        updated_at  TEXT,
                        is_deleted  INTEGER DEFAULT 0
                    )
                    """
                )
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
                # history 表 (对齐 mem0 SQLiteManager history 字段)
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id          TEXT PRIMARY KEY,
                        memory_id   TEXT,
                        old_memory  TEXT,
                        new_memory  TEXT,
                        event       TEXT,
                        created_at  TEXT,
                        is_deleted  INTEGER
                    )
                    """
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # ALTER TABLE 迁移 (代码内非 alembic, 幂等)
    # ------------------------------------------------------------------

    def _column_exists(self, table: str, column: str) -> bool:
        cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
        return column in cols

    def _migrate_add_state_columns(self) -> None:
        """添加 state/deleted_at/app_id 列 (幂等, ALTER TABLE)。"""
        with self._lock:
            if not self._column_exists("memories", "state"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN state TEXT DEFAULT 'active'")
            if not self._column_exists("memories", "deleted_at"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN deleted_at TEXT")
            if not self._column_exists("memories", "app_id"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN app_id TEXT")
            self.conn.commit()

    def _migrate_add_session_id_column(self) -> None:
        """添加 session_id 列 (幂等, ALTER TABLE)。"""
        with self._lock:
            if not self._column_exists("memories", "session_id"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")
            self.conn.commit()

    def _migrate_add_temporal_columns(self) -> None:
        """添加 valid_at/invalid_at/expired_at 列 (幂等, ALTER TABLE)。"""
        with self._lock:
            if not self._column_exists("memories", "valid_at"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN valid_at TEXT")
            if not self._column_exists("memories", "invalid_at"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN invalid_at TEXT")
            if not self._column_exists("memories", "expired_at"):
                self.conn.execute("ALTER TABLE memories ADD COLUMN expired_at TEXT")
            self.conn.commit()

    def _create_access_logs_table(self) -> None:
        """建 memory_access_logs 审计日志表。"""
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_access_logs (
                    id           TEXT PRIMARY KEY,
                    memory_id    TEXT NOT NULL,
                    app_id       TEXT,
                    access_type  TEXT NOT NULL,
                    metadata     TEXT,
                    accessed_at  TEXT NOT NULL
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id)")
            self.conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

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
        """添加记忆, 返回 memory_id (对齐 mem0 add 返回结构)。

        session_id: 会话 ID (对齐 mem0 run_id; None=不限制)。
        valid_at: 事实开始为真的时间 (双时态建模, 借鉴 graphiti EntityEdge)。
        """
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        meta = metadata or {}
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """
                    INSERT INTO memories
                        (id, user_id, agent_id, session_id, content, embedding, metadata,
                         created_at, updated_at, is_deleted, state, valid_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'active', ?)
                    """,
                    (
                        mid,
                        user_id,
                        agent_id,
                        session_id,
                        content,
                        json.dumps(embedding),
                        json.dumps(meta),
                        now,
                        now,
                        valid_at,
                    ),
                )
                self.conn.execute(
                    """
                    INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (str(uuid.uuid4()), mid, None, content, "ADD", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        # 双写: vector_store + keyword_index
        self._vector_store.insert_vectors([embedding], [mid], [{"user_id": user_id, "session_id": session_id}])
        self._keyword_index.add_docs({mid: content})
        logger.info("memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索 (numpy 余弦相似, embedder 已归一化则点积即余弦)。

        session_id: 仅搜该会话的记忆 (None=不限, 对齐 mem0 run_id)。
        """
        with self._lock:
            sql = "SELECT id, content, embedding, metadata, created_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            # mem0 风格 filters dict
            if filters:
                from septmuse.storage.filters import FiltersParser
                clean_filters = filters.copy()
                # 直接参数覆盖 filters 中的同名 key
                if session_id is not None:
                    clean_filters.pop("session_id", None)
                    clean_filters.pop("run_id", None)
                clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
                if clause:
                    sql += f" AND {clause}"
                    params.extend(fparams)
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm > 0:
            q = q / qnorm

        results: list[dict[str, Any]] = []
        for mid, content, emb_json, meta_json, created in rows:
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            score = float(np.dot(q, emb))
            if score >= threshold:
                results.append(
                    {
                        "id": mid,
                        "memory": content,
                        "score": score,
                        "metadata": json.loads(meta_json) if meta_json else {},
                        "created_at": created,
                    }
                )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部记忆 (对齐 mem0 get_all)。

        session_id: 仅返回该会话的记忆 (None=不限)。
        """
        with self._lock:
            sql = "SELECT id, content, metadata, created_at, updated_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            # mem0 风格 filters dict
            if filters:
                from septmuse.storage.filters import FiltersParser
                clean_filters = filters.copy()
                if session_id is not None:
                    clean_filters.pop("session_id", None)
                    clean_filters.pop("run_id", None)
                clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
                if clause:
                    sql += f" AND {clause}"
                    params.extend(fparams)
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
                "created_at": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条 (对齐 mem0 get)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, content, metadata, created_at FROM memories WHERE id=? AND is_deleted=0",
                (memory_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "memory": r[1],
            "metadata": json.loads(r[2]) if r[2] else {},
            "created_at": r[3],
        }

    def delete(self, memory_id: str) -> None:
        """软删除 (对齐 mem0 delete — 标记 is_deleted + state='deleted' + history)。"""
        now = _utcnow_iso()
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    "UPDATE memories SET is_deleted=1, state='deleted', deleted_at=?, updated_at=? WHERE id=?",
                    (now, now, memory_id),
                )
                self.conn.execute(
                    """
                    INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (str(uuid.uuid4()), memory_id, None, None, "DELETE", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        # 双写清理: vector_store + keyword_index
        self._vector_store.delete_vector(memory_id)
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
        """更新记忆 (对齐 mem0 update — UPDATE + history)。"""
        now = _utcnow_iso()
        with self._lock:
            cur = self.conn.execute(
                "SELECT content, metadata FROM memories WHERE id=? AND is_deleted=0",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            old_content, old_meta_json = row
            old_meta = json.loads(old_meta_json) if old_meta_json else {}

            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """UPDATE memories
                       SET content=?, embedding=?, metadata=?, updated_at=?
                       WHERE id=? AND is_deleted=0""",
                    (
                        content,
                        json.dumps(embedding),
                        json.dumps(metadata if metadata is not None else old_meta),
                        now,
                        memory_id,
                    ),
                )
                self.conn.execute(
                    """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (str(uuid.uuid4()), memory_id, old_content, content, "UPDATE", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        # 双写更新: vector_store + keyword_index
        self._vector_store.insert_vectors([embedding], [memory_id])
        self._keyword_index.add_docs({memory_id: content})
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        with self._lock:
            cur = self.conn.execute(
                """SELECT id, memory_id, old_memory, new_memory, event, created_at, is_deleted
                   FROM history WHERE memory_id=? ORDER BY created_at""",
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "is_deleted": bool(r[6]),
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------
    # 关系查询 (跨 agent 共享, 对齐 mem0 user_id 共享模式)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT DISTINCT agent_id FROM memories WHERE user_id=? AND is_deleted=0",
                (user_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows if r[0] is not None]

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT DISTINCT user_id FROM memories WHERE agent_id=? AND is_deleted=0",
                (agent_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, user_id, agent_id, content, metadata, created_at FROM memories "
                "WHERE user_id=? AND is_deleted=0 ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "agent_id": r[2],
                "memory": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
                "created_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 组合检索 (BM25 关键词 + 向量 RRF 融合)
    # ------------------------------------------------------------------

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25, 委托 _keyword_index)。

        session_id: 仅搜该会话的记忆 (None=不限)。
        """
        scores = self._keyword_index.retrieve(query, limit=top_k * 2)
        if not scores:
            return []

        mids = list(scores.keys())
        placeholders = ",".join("?" * len(mids))
        sql = (
            f"SELECT id, content, metadata, created_at FROM memories "
            f"WHERE id IN ({placeholders}) AND user_id=? AND is_deleted=0"
        )
        params: list[Any] = [*mids, user_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        with self._lock:
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            mid = r[0]
            results.append(
                {
                    "id": mid,
                    "memory": r[1],
                    "score": scores.get(mid, 0.0),
                    "metadata": json.loads(r[2]) if r[2] else {},
                    "created_at": r[3],
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
        """混合检索 (向量 + 关键词 RRF 融合, 对齐 base.hybrid_search)。"""
        vec_results = self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw_results = self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return _rrf_fuse(vec_results, kw_results, alpha=alpha)[:top_k]

    # ------------------------------------------------------------------
    # 访问日志 (审计)
    # ------------------------------------------------------------------

    def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志 (审计用)。"""
        log_id = str(uuid.uuid4())
        now = _utcnow_iso()
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory_access_logs (id, memory_id, app_id, access_type, metadata, accessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (log_id, memory_id, app_id, access_type, json.dumps(metadata) if metadata else None, now),
            )
            self.conn.commit()
        return log_id

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志 (按 accessed_at 降序)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, memory_id, app_id, access_type, metadata, accessed_at "
                "FROM memory_access_logs WHERE memory_id=? ORDER BY accessed_at DESC LIMIT ?",
                (memory_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "app_id": r[2],
                "access_type": r[3],
                "metadata": json.loads(r[4]) if r[4] else None,
                "accessed_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 时态 (双时态建模, 借鉴 graphiti EntityEdge)
    # ------------------------------------------------------------------

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 (valid_at <= t AND (invalid_at IS NULL OR invalid_at > t))。

        session_id: 仅返回该会话的记忆 (None=不限)。
        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
        """
        with self._lock:
            sql = (
                "SELECT id, content, metadata, created_at, valid_at, invalid_at "
                "FROM memories WHERE user_id=? AND is_deleted=0 "
                "AND (valid_at IS NULL OR valid_at <= ?) "
                "AND (invalid_at IS NULL OR invalid_at > ?)"
            )
            params: list[Any] = [user_id, reference_time, reference_time]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
                "created_at": r[3],
                "valid_at": r[4],
                "invalid_at": r[5],
            }
            for r in rows
        ]

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆 (借鉴 cognee temporal_retriever)。

        条件: valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)
        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
        """
        with self._lock:
            sql = (
                "SELECT id, content, metadata, created_at, valid_at, invalid_at "
                "FROM memories WHERE user_id=? AND is_deleted=0 "
                "AND (valid_at IS NULL OR valid_at <= ?) "
                "AND (invalid_at IS NULL OR invalid_at > ?)"
            )
            params: list[Any] = [user_id, end, start]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
                "created_at": r[3],
                "valid_at": r[4],
                "invalid_at": r[5],
            }
            for r in rows
        ]

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。

        Returns: {"id", "invalid_at", "expired_at", "event": "INVALIDATE"} or {"id", "event": "NOT_FOUND"}
        """
        existing = self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}

        inv_at = invalid_at or _utcnow_iso()
        exp_at = _utcnow_iso()
        now = _utcnow_iso()
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    "UPDATE memories SET invalid_at=?, expired_at=?, updated_at=? WHERE id=?",
                    (inv_at, exp_at, now, memory_id),
                )
                self.conn.execute(
                    """
                    INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (str(uuid.uuid4()), memory_id, existing.get("memory"), None, "INVALIDATE", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        logger.info("memory_invalidated", memory_id=memory_id, invalid_at=inv_at)
        return {"id": memory_id, "invalid_at": inv_at, "expired_at": exp_at, "event": "INVALIDATE"}

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
