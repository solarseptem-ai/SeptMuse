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
"""SQLite 图存储后端 — 零配置默认 GraphStore 实现。

复用 SQLiteMemoryStore 的 conn 和 _lock (同一 SQLite 文件), 管理 memory_links 表。
提取自原 concerns/evolution/zettel.py 的 _init_table/_create_link/get_links 逻辑。

参考模式:
- 表结构: 原 zettel.py memory_links (id/source_id/target_id/relation/score/created_at + UNIQUE)
- 幂等: INSERT OR IGNORE (UNIQUE 约束自动去重)
- 线程安全: 复用 SQLiteMemoryStore._lock
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.graph.base import GraphEdge, GraphStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteGraphStore(GraphStore):
    """SQLite 图存储 (零配置默认, 复用 SQLiteMemoryStore 连接)。

    用法:
        from septmuse.storage.sqlite.store import SQLiteMemoryStore
        store = SQLiteMemoryStore()
        graph = SQLiteGraphStore(store.conn, store._lock)
        graph.add_edge("mem-1", "mem-2", "related_to", 0.85)
        edges = graph.get_edges("mem-1")
    """

    def __init__(self, conn: Any, lock: threading.Lock) -> None:
        """共享 SQLiteMemoryStore 的 conn 和 _lock (同文件, 线程安全)。"""
        self.conn = conn
        self._lock = lock
        self._init_table()
        logger.info("sqlite_graph_store_ready")

    def _init_table(self) -> None:
        """建 memory_links 表 (提取自 zettel._init_table)。"""
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_links (
                    id          TEXT PRIMARY KEY,
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    relation    TEXT NOT NULL DEFAULT 'related_to',
                    score       REAL NOT NULL DEFAULT 0.0,
                    created_at  TEXT,
                    UNIQUE(source_id, target_id, relation)
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)")

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        score: float = 0.0,
    ) -> str:
        """添加有向边, 幂等 (INSERT OR IGNORE + UNIQUE 约束)。返回 edge_id。"""
        edge_id = f"link-{uuid.uuid4()}"
        now = _utcnow_iso()
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO memory_links (id, source_id, target_id, relation, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (edge_id, source_id, target_id, relation, score, now),
            )
        return edge_id

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """获取出边 (source_id == node_id)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, source_id, target_id, relation, score FROM memory_links WHERE source_id=?",
                (node_id,),
            )
            rows = cur.fetchall()
        return [GraphEdge(id=r[0], source_id=r[1], target_id=r[2], relation=r[3], score=r[4]) for r in rows]

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        """获取邻居节点 ID 列表。relation=None 表示所有关系。"""
        with self._lock:
            if relation is None:
                cur = self.conn.execute(
                    "SELECT target_id FROM memory_links WHERE source_id=?",
                    (node_id,),
                )
            else:
                cur = self.conn.execute(
                    "SELECT target_id FROM memory_links WHERE source_id=? AND relation=?",
                    (node_id, relation),
                )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        """检查边是否存在。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT 1 FROM memory_links WHERE source_id=? AND target_id=? AND relation=?",
                (source_id, target_id, relation),
            )
            return cur.fetchone() is not None

    def delete_edge(self, edge_id: str) -> bool:
        """删除边。True=删除成功, False=不存在 (基于 DELETE rowcount)。"""
        with self._lock:
            cur = self.conn.execute("DELETE FROM memory_links WHERE id = ?", (edge_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        """无独立资源 (conn 由 SQLiteMemoryStore 管理, 不关闭)。"""
        pass
