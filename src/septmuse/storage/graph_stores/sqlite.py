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

复用 ORMMemoryStore 的 raw_connection 和 lock (同一 SQLite 文件), 管理 memory_links + memory_nodes 表。

参考模式:
- 边表: memory_links (id/source_id/target_id/relation/score/created_at + UNIQUE)
- 节点表: memory_nodes (id/properties JSON/created_at)
- 幂等: INSERT OR IGNORE (边) / INSERT OR REPLACE (节点)
- 多跳遍历: WITH RECURSIVE CTE (单次 SQL 完成多跳, 比 Python 循环快)
- 社区检测: 纯 Python label_propagation (借鉴 graphiti)
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.graph_stores.base import GraphEdge, GraphStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteGraphStore(GraphStore):
    """SQLite 图存储 (零配置默认, 复用 ORMMemoryStore raw_connection)。

    用法:
        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
        store = ORMMemoryStore(engine)
        raw_conn = store.engine.raw_connection()
        graph = SQLiteGraphStore(raw_conn, threading.Lock())
        graph.add_edge("mem-1", "mem-2", "related_to", 0.85)
        edges = graph.get_edges("mem-1")
    """

    def __init__(self, conn: Any, lock: threading.Lock) -> None:
        """共享 ORMMemoryStore 的 raw_connection 和 lock (同文件, 线程安全)。"""
        self.conn = conn
        self._lock = lock
        self._init_table()
        logger.info("sqlite_graph_store_ready")

    def _init_table(self) -> None:
        """建 memory_links + memory_nodes 表。"""
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
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id          TEXT PRIMARY KEY,
                    properties  TEXT DEFAULT '{}',
                    created_at  TEXT
                )
                """
            )

    # ── 边管理 (原有) ──

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
            self.conn.commit()
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

    # ── 节点管理 (新增) ──

    def add_node(self, node_id: str, properties: dict[str, Any] | None = None) -> None:
        """添加或更新节点 (幂等, INSERT OR REPLACE)。"""
        now = _utcnow_iso()
        props = json.dumps(properties or {}, ensure_ascii=False)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_nodes (id, properties, created_at) VALUES (?, ?, ?)",
                (node_id, props, now),
            )
            self.conn.commit()

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """获取节点属性。None=不存在。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT properties, created_at FROM memory_nodes WHERE id=?",
                (node_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"id": node_id, "properties": json.loads(row[0]), "created_at": row[1]}

    def delete_node(self, node_id: str, *, cascade: bool = True) -> bool:
        """删除节点。cascade=True 同时删除关联边。"""
        with self._lock:
            cur = self.conn.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
            deleted = cur.rowcount > 0
            if cascade and deleted:
                self.conn.execute(
                    "DELETE FROM memory_links WHERE source_id=? OR target_id=?",
                    (node_id, node_id),
                )
            self.conn.commit()
            return deleted

    # ── 入边查询 (新增) ──

    def get_in_edges(self, node_id: str) -> list[GraphEdge]:
        """获取节点的所有入边 (target_id == node_id)。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, source_id, target_id, relation, score FROM memory_links WHERE target_id=?",
                (node_id,),
            )
            rows = cur.fetchall()
        return [GraphEdge(id=r[0], source_id=r[1], target_id=r[2], relation=r[3], score=r[4]) for r in rows]

    # ── 图统计 (新增) ──

    def get_stats(self) -> dict[str, Any]:
        """图统计: node_count, edge_count, density。"""
        with self._lock:
            cur_n = self.conn.execute("SELECT COUNT(*) FROM memory_nodes")
            node_count = cur_n.fetchone()[0]
            cur_e = self.conn.execute("SELECT COUNT(*) FROM memory_links")
            edge_count = cur_e.fetchone()[0]
        max_edges = node_count * (node_count - 1) if node_count > 1 else 1
        density = (edge_count / max_edges) if max_edges > 0 else 0.0
        return {"node_count": node_count, "edge_count": edge_count, "density": round(density, 4)}

    # ── 多跳遍历 (override: 递归 CTE) ──

    def traverse(
        self,
        seed_id: str,
        max_depth: int = 2,
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """BFS 遍历 (递归 CTE, 单次 SQL 完成多跳)。

        direction: "out"=出边, "in"=入边, "both"=双向。
        """
        if max_depth < 1:
            return []

        col = "target_id" if direction == "out" else "source_id"
        filter_col = "source_id" if direction == "out" else "target_id"

        rel_clause = f"AND relation = '{relation}'" if relation else ""

        sql = f"""
        WITH RECURSIVE bfs(node_id, depth) AS (
            SELECT {col}, 1 FROM memory_links
            WHERE {filter_col} = ? {rel_clause}
            UNION
            SELECT l.{col}, b.depth + 1 FROM memory_links l
            JOIN bfs b ON l.{filter_col} = b.node_id
            WHERE b.depth < ? {rel_clause}
        )
        SELECT node_id, MIN(depth) as depth FROM bfs
        WHERE node_id != ?
        GROUP BY node_id ORDER BY depth
        """

        if direction == "both":
            return self._traverse_both(seed_id, max_depth, relation)

        with self._lock:
            cur = self.conn.execute(sql, (seed_id, max_depth, seed_id))
            rows = cur.fetchall()
        return [{"id": r[0], "depth": r[1]} for r in rows]

    def _traverse_both(self, seed_id: str, max_depth: int, relation: str | None) -> list[dict[str, Any]]:
        """双向遍历 (出边 + 入边, 去重)。"""
        out = self.traverse(seed_id, max_depth, direction="out", relation=relation)
        in_ = self.traverse(seed_id, max_depth, direction="in", relation=relation)
        seen: dict[str, int] = {}
        for item in out + in_:
            nid = item["id"]
            if nid not in seen or item["depth"] < seen[nid]:
                seen[nid] = item["depth"]
        return [{"id": nid, "depth": d} for nid, d in sorted(seen.items(), key=lambda x: x[1])]

    # ── 社区检测 (override: 纯 Python label_propagation) ──

    def detect_communities(self, algorithm: str = "label_propagation") -> dict[str, list[str]]:
        """社区检测 — 纯 Python label_propagation (借鉴 graphiti)。

        算法: 每个节点初始社区为自己, 迭代将节点分配到邻居中最多的社区, 直到收敛。
        """
        if algorithm != "label_propagation":
            raise NotImplementedError(f"Algorithm '{algorithm}' not supported")

        with self._lock:
            cur = self.conn.execute("SELECT DISTINCT source_id FROM memory_links UNION SELECT DISTINCT target_id FROM memory_links")
            node_ids = [r[0] for r in cur.fetchall()]
            cur = self.conn.execute("SELECT source_id, target_id FROM memory_links")
            edge_list = [(r[0], r[1]) for r in cur.fetchall()]

        if not node_ids:
            return {}

        adj: dict[str, set[str]] = defaultdict(set)
        for s, t in edge_list:
            adj[s].add(t)
            adj[t].add(s)

        labels: dict[str, str] = {n: n for n in node_ids}

        changed = True
        iterations = 0
        max_iterations = len(node_ids) * 2
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for node in node_ids:
                neighbor_labels: dict[str, int] = defaultdict(int)
                for neighbor in adj.get(node, set()):
                    neighbor_labels[labels[neighbor]] += 1
                if not neighbor_labels:
                    continue
                best_label = max(neighbor_labels, key=neighbor_labels.get)
                if best_label != labels[node]:
                    labels[node] = best_label
                    changed = True

        communities: dict[str, list[str]] = defaultdict(list)
        for node, label in labels.items():
            communities[label].append(node)

        return dict(communities)

    def close(self) -> None:
        """无独立资源 (conn 由 ORMMemoryStore 管理, 不关闭)。"""
        pass
