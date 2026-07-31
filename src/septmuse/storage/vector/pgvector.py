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
"""PostgreSQL + pgvector 记忆存储后端 — SeptMuse 生产主选 (架构文档 §3 storage/vector/pgvector.py)。

组合 memories + history + memory_access_logs 于 Postgres, 向量检索用 pgvector
扩展的 <=> 余弦距离 (score = max(0, 1-distance) 归一化)。

参考模式 (实证, 非自行发挥):
- 连接池/SQL: mem0/mem0/vector_stores/pgvector.py (psycopg3 优先, psycopg2 回退)
- add/search/get_all/delete 契约: mem0 Memory 方法签名
- history 表字段: mem0 SQLiteManager history 表 (memory_id/old_memory/new_memory/event/created_at/is_deleted)
- 组合 BM25: SQLiteMemoryStore Task 5 模式 (PG 无内置 BM25, 用 SQLiteBM25Index 辅助)
- 权限层: SQLiteMemoryStore P2 Task 3 模式 (state 状态机 + memory_access_logs 审计表)
"""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)

# psycopg3 优先, psycopg2 回退 (对齐 mem0 PGVector + AGEGraphStore 模式)
try:
    from psycopg.types.json import Json
    from psycopg_pool import ConnectionPool

    PSYCOPG_VERSION = 3
except ImportError:
    try:
        from psycopg2.extras import Json
        from psycopg2.pool import ThreadedConnectionPool as ConnectionPool

        PSYCOPG_VERSION = 2
    except ImportError as _e:
        raise ImportError(
            "Neither 'psycopg' nor 'psycopg2' is available. "
            "Install with: pip install psycopg[pool] or pip install psycopg2"
        ) from _e


def _utcnow_iso() -> str:
    """UTC ISO 时间戳 (对齐 SQLiteMemoryStore)。"""
    return datetime.now(timezone.utc).isoformat()


def _to_pgvector_str(vec: list[float]) -> str:
    """将向量转为 pgvector 字符串字面量 (psycopg2 不原生适配 vector 类型)。

    >>> _to_pgvector_str([0.1, 0.2, 0.3])
    '[0.1,0.2,0.3]'
    """
    return "[" + ",".join(str(v) for v in vec) + "]"


def _with_sslmode(connection_string: str, sslmode: str) -> str:
    """在 URI 或 keyword conninfo 中添加/替换 sslmode 参数 (对齐 mem0 PGVector)。

    支持两种格式:
    - URI: postgresql://user:pass@host:5432/db?sslmode=require
    - conninfo: host=localhost dbname=test sslmode=require
    """
    if "://" in connection_string:
        parsed = urlsplit(connection_string)
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "sslmode"]
        query.append(("sslmode", sslmode))
        return parsed._replace(query=urlencode(query)).geturl()

    if re.search(r"(^|\s)sslmode=", connection_string):
        return re.sub(r"(^|\s)sslmode=\S+", lambda m: f"{m.group(1)}sslmode={sslmode}", connection_string)

    return f"{connection_string} sslmode={sslmode}"


def _parse_jsonb(val: Any) -> dict[str, Any]:
    """解析 JSONB 返回值 (psycopg3 自动反序列化为 dict, psycopg2 返回 str, 兼容两者)。"""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return json.loads(val) if val else {}
    return val


class PGVectorStore(MemoryStore):
    """PostgreSQL + pgvector 记忆存储后端 (实现 MemoryStore ABC)。

    零配置不可用 — 需 Postgres + pgvector 扩展。
    用法:
        store = PGVectorStore(connection_string="postgresql://user:pass@host:5432/db")
        mid = store.add("hello", [0.1, ...], user_id="alice")
        results = store.search([0.1, ...], user_id="alice", top_k=5)

    组合 (P1 Task 6): SQLiteBM25Index 辅助 BM25 关键词检索 (PG 无内置 BM25)。
    权限层 (P2 Task 7): state 状态机 + memory_access_logs 审计表。
    """

    def __init__(
        self,
        *,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        connection_string: str | None = None,
        connection_pool: Any | None = None,
        embedding_model_dims: int = 1536,
        collection_name: str = "memories",
        minconn: int = 1,
        maxconn: int = 5,
        sslmode: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._dim = embedding_model_dims
        self._collection_ensured = False

        # 连接池: 外部传入优先, 否则自建 (对齐 AGEGraphStore 模式)
        if connection_pool is not None:
            self.connection_pool: Any = connection_pool
        else:
            conn_str = connection_string or f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            if sslmode:
                conn_str = _with_sslmode(conn_str, sslmode)
            if PSYCOPG_VERSION == 3:
                self.connection_pool = ConnectionPool(conninfo=conn_str, min_size=minconn, max_size=maxconn, open=False)
                self.connection_pool.open(wait=False)
            else:
                self.connection_pool = ConnectionPool(minconn=minconn, maxconn=maxconn, dsn=conn_str)

        # P1 Task 6: SQLiteBM25Index 辅助 BM25 (PG 无内置 BM25, 与 PG 数据独立)
        from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index

        self._keyword_index = SQLiteBM25Index(db_path=Path(tempfile.gettempdir()) / "septmuse_pg_bm25.db")

        logger.info("pgvector_store_ready", collection=collection_name, dims=embedding_model_dims)

    def _ensure_collection(self) -> None:
        """首次操作时 lazy 建表 (CREATE EXTENSION + CREATE TABLE + ALTER TABLE)。"""
        if self._collection_ensured:
            return
        self._create_tables()
        self._collection_ensured = True

    def _create_tables(self) -> None:
        """建表 + 扩展 + P2 权限列迁移 (ALTER TABLE IF NOT EXISTS, PG 9.6+)。"""
        with self._get_cursor(commit=True) as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.collection_name} ("
                "id TEXT PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "agent_id TEXT, "
                "content TEXT NOT NULL, "
                f"embedding vector({self._dim}), "
                "metadata JSONB, "
                "created_at TEXT, "
                "updated_at TEXT, "
                "is_deleted INTEGER DEFAULT 0"
                ")"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_user ON {self.collection_name}(user_id)"
            )
            # history 表 (对齐 mem0 SQLiteManager history 字段)
            cur.execute(
                "CREATE TABLE IF NOT EXISTS history ("
                "id TEXT PRIMARY KEY, "
                "memory_id TEXT, "
                "old_memory TEXT, "
                "new_memory TEXT, "
                "event TEXT, "
                "created_at TEXT, "
                "is_deleted INTEGER"
                ")"
            )
            # P2 Task 7: state 状态机列 (ALTER TABLE IF NOT EXISTS, PG 9.6+)
            # DEFAULT 'active' 会使既有行自动回填为 'active' (PG ALTER 带 DEFAULT 行为)
            cur.execute(f"ALTER TABLE {self.collection_name} ADD COLUMN IF NOT EXISTS state TEXT DEFAULT 'active'")
            cur.execute(f"ALTER TABLE {self.collection_name} ADD COLUMN IF NOT EXISTS app_id TEXT")
            cur.execute(f"ALTER TABLE {self.collection_name} ADD COLUMN IF NOT EXISTS archived_at TEXT")
            cur.execute(f"ALTER TABLE {self.collection_name} ADD COLUMN IF NOT EXISTS deleted_at TEXT")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_user_state "
                f"ON {self.collection_name}(user_id, state)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_app_state "
                f"ON {self.collection_name}(app_id, state)"
            )
            # P2 Task 7: memory_access_logs 审计表 (借鉴 mem0 models.py)
            cur.execute(
                "CREATE TABLE IF NOT EXISTS memory_access_logs ("
                "id TEXT PRIMARY KEY, "
                "memory_id TEXT NOT NULL, "
                "app_id TEXT, "
                "accessed_at TEXT NOT NULL, "
                "access_type TEXT NOT NULL, "
                "metadata JSONB DEFAULT '{}'::jsonb"
                ")"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_memory_time ON memory_access_logs(memory_id, accessed_at)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_access_app_time ON memory_access_logs(app_id, accessed_at)")

    @contextlib.contextmanager
    def _get_cursor(self, commit: bool = False) -> Iterator[Any]:
        """统一 cursor contextmanager (对齐 mem0 PGVector._get_cursor + AGEGraphStore)。

        psycopg3 自动管理 commit/rollback + 连接归还;
        psycopg2 手动 getconn/putconn。
        """
        if PSYCOPG_VERSION == 3:
            with self.connection_pool.connection() as conn, conn.cursor() as cur:
                try:
                    yield cur
                    if commit:
                        conn.commit()
                except Exception:
                    conn.rollback()
                    logger.error("pg_cursor_error", exc_info=True)
                    raise
        else:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
                self.connection_pool.putconn(conn)

    # ------------------------------------------------------------------
    # P2 Task 7: 访问日志 (权限层审计)
    # ------------------------------------------------------------------

    def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """记录记忆访问日志 (供 governance.access_log.record_access 委托调用)。

        Returns: log_id (格式 log-{uuid})
        """
        self._ensure_collection()
        log_id = f"log-{uuid.uuid4()}"
        with self._get_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO memory_access_logs (id, memory_id, app_id, accessed_at, access_type, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (log_id, memory_id, app_id, _utcnow_iso(), access_type, Json(metadata or {})),
            )
        return log_id

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """读取指定记忆的访问日志, 按 accessed_at DESC 排序 (最新在前)。

        覆盖 MemoryStore 默认空实现; PG/psycopg 自动反序列化 JSONB。
        """
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                "SELECT id, memory_id, app_id, accessed_at, access_type, metadata "
                "FROM memory_access_logs WHERE memory_id = %s ORDER BY accessed_at DESC LIMIT %s",
                (memory_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "app_id": r[2],
                "accessed_at": r[3],
                "access_type": r[4],
                "metadata": _parse_jsonb(r[5]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # MemoryStore ABC 实现
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆, 返回 memory_id (对齐 mem0 add)。

        valid_at 暂未持久化 (pgvector 无 temporal 列, 向后兼容接受)。
        """
        self._ensure_collection()
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        with self._get_cursor(commit=True) as cur:
            cur.execute(
                f"INSERT INTO {self.collection_name} "
                "(id, user_id, agent_id, content, embedding, metadata, created_at, updated_at, is_deleted) "
                "VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, 0)",
                (
                    mid,
                    user_id,
                    agent_id,
                    content,
                    _to_pgvector_str(embedding),
                    Json(metadata or {}),
                    now,
                    now,
                ),
            )
            cur.execute(
                "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted) "
                "VALUES (%s, %s, %s, %s, %s, %s, 0)",
                (str(uuid.uuid4()), mid, None, content, "ADD", now),
            )
        # P1 Task 6: best-effort keyword index dual-write (PG 主存储, 失败仅 warning)
        with contextlib.suppress(Exception):
            self._keyword_index.add_docs({mid: content})
        logger.info("memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索 (pgvector <=> 余弦距离, score = max(0, 1-distance))。

        P2 Task 7: WHERE 增加 state='active' OR state IS NULL 过滤。
        """
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, created_at, "
                f"embedding <=> %s::vector AS distance "
                f"FROM {self.collection_name} "
                "WHERE user_id = %s AND is_deleted = 0 "
                "AND (state = 'active' OR state IS NULL) "
                "ORDER BY distance LIMIT %s",
                (_to_pgvector_str(query_embedding), user_id, top_k),
            )
            rows = cur.fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            distance = float(r[4])
            score = max(0.0, 1.0 - distance)
            if score < threshold:
                continue
            results.append(
                {
                    "id": r[0],
                    "memory": r[1],
                    "metadata": _parse_jsonb(r[2]),
                    "created_at": r[3],
                    "score": score,
                }
            )
        return results

    def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆 (P2: 仅 active 状态)。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, created_at, updated_at "
                f"FROM {self.collection_name} "
                "WHERE user_id = %s AND is_deleted = 0 "
                "AND (state = 'active' OR state IS NULL)",
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "metadata": _parse_jsonb(r[2]),
                "created_at": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条 (对齐 mem0 get)。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, created_at "
                f"FROM {self.collection_name} "
                "WHERE id = %s AND is_deleted = 0",
                (memory_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "memory": r[1],
            "metadata": _parse_jsonb(r[2]),
            "created_at": r[3],
        }

    def delete(self, memory_id: str) -> None:
        """软删除 (对齐 mem0 delete — is_deleted=1 + state='deleted' + deleted_at + history)。

        P2 Task 7: UPDATE 增加 state='deleted', deleted_at。
        P1 Task 6: best-effort keyword index 清理。
        """
        self._ensure_collection()
        now = _utcnow_iso()
        with self._get_cursor(commit=True) as cur:
            cur.execute(
                f"UPDATE {self.collection_name} "
                "SET is_deleted = 1, state = 'deleted', deleted_at = %s, updated_at = %s "
                "WHERE id = %s",
                (now, now, memory_id),
            )
            cur.execute(
                "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted) "
                "VALUES (%s, %s, %s, %s, %s, %s, 1)",
                (str(uuid.uuid4()), memory_id, None, None, "DELETE", now),
            )
        # P1 Task 6: best-effort keyword index 清理 (老记忆无条目静默)
        with contextlib.suppress(Exception):
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
        """更新记忆 (对齐 mem0 update — UPDATE + history + 同步关键词索引)。

        Returns: True=更新成功; False=memory_id 不存在或已删除。
        """
        self._ensure_collection()
        now = _utcnow_iso()
        with self._get_cursor(commit=True) as cur:
            cur.execute(
                f"SELECT content, metadata, user_id FROM {self.collection_name} WHERE id = %s AND is_deleted = 0",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            old_content, old_meta_raw, _mem_user_id = row
            old_meta = _parse_jsonb(old_meta_raw)

            cur.execute(
                f"UPDATE {self.collection_name} "
                "SET content = %s, embedding = %s::vector, metadata = %s, updated_at = %s "
                "WHERE id = %s AND is_deleted = 0",
                (
                    content,
                    _to_pgvector_str(embedding),
                    Json(metadata if metadata is not None else old_meta),
                    now,
                    memory_id,
                ),
            )
            cur.execute(
                "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted) "
                "VALUES (%s, %s, %s, %s, %s, %s, 0)",
                (str(uuid.uuid4()), memory_id, old_content, content, "UPDATE", now),
            )
        # P1 Task 6: best-effort keyword index 更新 (幂等覆盖, INSERT OR REPLACE)
        with contextlib.suppress(Exception):
            self._keyword_index.add_docs({memory_id: content})
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                "SELECT id, memory_id, old_memory, new_memory, event, created_at, is_deleted "
                "FROM history WHERE memory_id = %s ORDER BY created_at",
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
        """释放连接资源 (keyword_index 独立 conn 静默关闭 + pool close)。"""
        # P1 Task 6: keyword_index 独立 conn, 先关 (静默, contextlib.suppress 满足 ruff SIM105)
        with contextlib.suppress(Exception):
            self._keyword_index.close()
        # pool close (psycopg3: close(), psycopg2: closeall(); if 分支使 ruff SIM105 不报)
        try:
            if PSYCOPG_VERSION == 3:
                self.connection_pool.close()
            else:
                self.connection_pool.closeall()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 关系查询 (跨 agent 共享, 对齐 SQLiteMemoryStore)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT agent_id FROM {self.collection_name} WHERE user_id = %s AND is_deleted = 0",
                (user_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows if r[0] is not None]

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT user_id FROM {self.collection_name} WHERE agent_id = %s AND is_deleted = 0",
                (agent_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        self._ensure_collection()
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT id, user_id, agent_id, content, metadata, created_at "
                f"FROM {self.collection_name} "
                "WHERE user_id = %s AND is_deleted = 0 ORDER BY created_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "agent_id": r[2],
                "memory": r[3],
                "metadata": _parse_jsonb(r[4]),
                "created_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # P1 Task 6: 关键词检索 (BM25, 委托 SQLiteBM25Index + JOIN memories)
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """关键词检索 (BM25, 委托 SQLiteBM25Index + JOIN memories)。

        PG 无内置 BM25, 用 SQLiteBM25Index 辅助索引 (与 PG 数据独立, 双写维护)。
        返回格式同 search: [{"id", "memory", "score", "metadata", "created_at"}]
        """
        self._ensure_collection()
        kw_scores = self._keyword_index.retrieve(query, limit=top_k * 2)
        if not kw_scores:
            return []
        results: list[dict[str, Any]] = []
        for doc_id, score in kw_scores.items():
            with self._get_cursor() as cur:
                cur.execute(
                    f"SELECT content, metadata, created_at, user_id "
                    f"FROM {self.collection_name} "
                    "WHERE id = %s AND is_deleted = 0",
                    (doc_id,),
                )
                row = cur.fetchone()
            if row and row[3] == user_id:
                content, metadata_json, created_at, _ = row
                results.append(
                    {
                        "id": doc_id,
                        "memory": content,
                        "score": score,
                        "metadata": _parse_jsonb(metadata_json),
                        "created_at": created_at,
                    }
                )
        return results[:top_k]

    def __del__(self) -> None:
        """析构时静默关闭 (对齐 SQLiteMemoryStore.__del__)。"""
        with contextlib.suppress(Exception):
            self.close()
