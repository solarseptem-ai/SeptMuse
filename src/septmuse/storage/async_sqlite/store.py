"""异步 SQLite 记忆存储（aiosqlite）。

表结构与 sync SQLiteMemoryStore 一致，同一个 DB 文件可共享。
双写组件（SQLiteVectorStore + SQLiteBM25Index）用 asyncio.to_thread 包装。
score 语义: 相似度 (越高越相似, 范围 [0, 1])。
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from septmuse.core.logging import get_logger
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index
from septmuse.storage.vector.sqlite_vec import SQLiteVectorStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class AsyncSQLiteMemoryStore(AsyncMemoryStore):
    """异步 SQLite 记忆存储（aiosqlite）。

    用法:
        store = AsyncSQLiteMemoryStore(db_path="mem.db")
        mid = await store.add("hello", [1.0, ...], user_id="alice")
        results = await store.search([1.0, ...], user_id="alice")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".septmuse" / "septmuse.db"
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._vector_store: SQLiteVectorStore | None = None
        self._keyword_index: SQLiteBM25Index | None = None

    async def _ensure_conn(self) -> aiosqlite.Connection:
        """延迟打开连接（首次操作时）。"""
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
            await self._create_tables()
            await asyncio.to_thread(self._init_dual_write)
        return self._conn

    def _init_dual_write(self) -> None:
        """初始化双写组件（sync，在 to_thread 中调用）。"""
        sync_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._vector_store = SQLiteVectorStore(conn=sync_conn)
        self._keyword_index = SQLiteBM25Index(db_path=self._db_path)
        # 轻量级迁移：在 sync 连接上运行（DDL，快，一次性）
        from septmuse.storage.migrations.runner import MigrationRunner
        MigrationRunner(sync_conn, "sqlite").run()

    async def _create_tables(self) -> None:
        """建表（与 sync 版 DDL 一致）。"""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                agent_id    TEXT,
                session_id  TEXT,
                content     TEXT NOT NULL,
                embedding   TEXT,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT,
                updated_at  TEXT,
                valid_at    TEXT,
                invalid_at  TEXT,
                expired_at  TEXT,
                is_deleted  INTEGER DEFAULT 0,
                state       TEXT DEFAULT 'active',
                app_id      TEXT,
                archived_at TEXT,
                deleted_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE TABLE IF NOT EXISTS history (
                id          TEXT PRIMARY KEY,
                memory_id   TEXT,
                old_memory  TEXT,
                new_memory  TEXT,
                event       TEXT,
                created_at  TEXT,
                is_deleted  INTEGER
            );
            CREATE TABLE IF NOT EXISTS memory_access_logs (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT NOT NULL,
                app_id       TEXT,
                access_type  TEXT NOT NULL,
                metadata     TEXT,
                accessed_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id);
        """)
        await self._conn.commit()

    # ── CRUD ──

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
        conn = await self._ensure_conn()
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        await conn.execute(
            """INSERT INTO memories (id, user_id, agent_id, session_id, content, embedding,
               metadata, created_at, updated_at, valid_at, is_deleted, state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'active')""",
            (mid, user_id, agent_id, session_id, content, json.dumps(embedding),
             json.dumps(metadata or {}), now, now, valid_at),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), mid, None, content, "ADD", now),
        )
        await conn.commit()
        if self._vector_store:
            await asyncio.to_thread(
                self._vector_store.insert_vectors, [embedding], [mid], [{"user_id": user_id}]
            )
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.add_docs, {mid: content})
        logger.info("async_memory_added", memory_id=mid, user_id=user_id)
        return mid

    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_conn()
        sql = """SELECT id, content, metadata, created_at, embedding FROM memories
                 WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)"""
        params_list: list[Any] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            params_list.append(session_id)
        if filters:
            from septmuse.storage.filters import FiltersParser
            clean_filters = filters.copy()
            if session_id:
                clean_filters.pop("session_id", None)
                clean_filters.pop("run_id", None)
            clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
            if clause:
                sql += f" AND {clause}"
                params_list.extend(fparams)
        cursor = await conn.execute(sql, params_list)
        rows = await cursor.fetchall()
        scored = await asyncio.to_thread(self._score_rows, query_embedding, rows)
        return [r for r in scored if r["score"] >= threshold][:top_k]

    def _score_rows(self, query_embedding: list[float], rows: list[Any]) -> list[dict[str, Any]]:
        """余弦相似度计算（纯 CPU，无 I/O）。"""
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []
        scored: list[dict[str, Any]] = []
        for row in rows:
            vid, content, meta_json, created_at, emb_json = row
            vec = np.array(json.loads(emb_json), dtype=np.float32)
            if vec.shape != query.shape:
                continue
            vec_norm = float(np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            score = max(0.0, min(1.0, score))
            scored.append({
                "id": vid, "memory": content, "score": score,
                "metadata": json.loads(meta_json) if meta_json else {},
                "created_at": created_at,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_conn()
        sql = """SELECT id, content, metadata, created_at, updated_at FROM memories
                 WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)"""
        params_list: list[Any] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            params_list.append(session_id)
        if filters:
            from septmuse.storage.filters import FiltersParser
            clean_filters = filters.copy()
            if session_id:
                clean_filters.pop("session_id", None)
                clean_filters.pop("run_id", None)
            clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
            if clause:
                sql += f" AND {clause}"
                params_list.extend(fparams)
        cursor = await conn.execute(sql, params_list)
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "memory": r[1], "metadata": json.loads(r[2]) if r[2] else {},
             "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT id, content, metadata, created_at FROM memories
               WHERE id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
            (memory_id,),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return {
            "id": r[0], "memory": r[1],
            "metadata": json.loads(r[2]) if r[2] else {}, "created_at": r[3],
        }

    async def delete(self, memory_id: str) -> None:
        conn = await self._ensure_conn()
        now = _utcnow_iso()
        await conn.execute(
            """UPDATE memories SET is_deleted=1, state='deleted', deleted_at=?, updated_at=? WHERE id=?""",
            (now, now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (str(uuid.uuid4()), memory_id, None, None, "DELETE", now),
        )
        await conn.commit()
        if self._vector_store:
            await asyncio.to_thread(self._vector_store.delete_vector, memory_id)
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.delete_docs, [memory_id])
        logger.info("async_memory_deleted", memory_id=memory_id)

    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        conn = await self._ensure_conn()
        now = _utcnow_iso()
        cursor = await conn.execute(
            "SELECT content, metadata FROM memories WHERE id=? AND is_deleted=0",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        old_content, old_meta_json = row
        old_meta = json.loads(old_meta_json) if old_meta_json else {}
        await conn.execute(
            """UPDATE memories SET content=?, embedding=?, metadata=?, updated_at=? WHERE id=? AND is_deleted=0""",
            (content, json.dumps(embedding), json.dumps(metadata if metadata is not None else old_meta), now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), memory_id, old_content, content, "UPDATE", now),
        )
        await conn.commit()
        if self._vector_store:
            await asyncio.to_thread(
                self._vector_store.insert_vectors, [embedding], [memory_id], [{"user_id": ""}]
            )
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.add_docs, {memory_id: content})
        logger.info("async_memory_updated", memory_id=memory_id)
        return True

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT id, memory_id, old_memory, new_memory, event, created_at, is_deleted
               FROM history WHERE memory_id=? ORDER BY created_at""",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "memory_id": r[1], "old_memory": r[2], "new_memory": r[3],
             "event": r[4], "created_at": r[5], "is_deleted": bool(r[6])}
            for r in rows
        ]

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索（BM25，双写组件已初始化时生效）。"""
        if not self._keyword_index:
            return []
        # BM25 检索（sync，用 to_thread）
        scores = await asyncio.to_thread(self._keyword_index.retrieve, query, top_k * 2)
        if not scores:
            return []
        # 过滤 user_id（BM25 不区分用户，需要在 memories 表中验证）
        conn = await self._ensure_conn()
        results: list[dict[str, Any]] = []
        for doc_id, score in scores.items():
            cursor = await conn.execute(
                """SELECT content, metadata, created_at FROM memories
                   WHERE id=? AND user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
                (doc_id, user_id),
            )
            r = await cursor.fetchone()
            if r:
                results.append({
                    "id": doc_id, "memory": r[0], "score": float(score),
                    "metadata": json.loads(r[1]) if r[1] else {}, "created_at": r[2],
                })
        return results[:top_k]

    async def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志（审计用）。"""
        conn = await self._ensure_conn()
        log_id = str(uuid.uuid4())
        now = _utcnow_iso()
        await conn.execute(
            "INSERT INTO memory_access_logs (id, memory_id, app_id, access_type, metadata, accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, memory_id, app_id, access_type, json.dumps(metadata) if metadata else None, now),
        )
        await conn.commit()
        return log_id

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志（按 accessed_at 降序）。"""
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT id, memory_id, app_id, access_type, metadata, accessed_at "
            "FROM memory_access_logs WHERE memory_id=? ORDER BY accessed_at DESC LIMIT ?",
            (memory_id, limit),
        )
        rows = await cursor.fetchall()
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

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真（设置 invalid_at + expired_at，不删除记忆）。"""
        from datetime import datetime, timezone
        conn = await self._ensure_conn()
        existing = await self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        inv_at = invalid_at or datetime.now(timezone.utc).isoformat()
        exp_at = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE memories SET invalid_at=?, expired_at=?, updated_at=? WHERE id=?",
            (inv_at, exp_at, now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), memory_id, existing.get("memory"), None, "INVALIDATE", now),
        )
        await conn.commit()
        logger.info("async_memory_invalidated", memory_id=memory_id, invalid_at=inv_at)
        return {"id": memory_id, "invalid_at": inv_at, "expired_at": exp_at, "event": "INVALIDATE"}

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._vector_store:
            self._vector_store.close()
            self._vector_store = None
        if self._keyword_index:
            self._keyword_index.close()
            self._keyword_index = None
