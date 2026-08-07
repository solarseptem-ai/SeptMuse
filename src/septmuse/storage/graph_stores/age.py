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
"""Apache AGE 图存储后端 — SeptMuse 生产图主选 (架构文档 §3 storage/graph/age.py)。

复用 solarseptem 平台 Postgres, 通过 Apache AGE 扩展执行 Cypher 查询。
实现 GraphStore ABC, 接口与 SQLiteGraphStore 完全对齐 (可插拔替换)。

参考模式 (实证, 非自行发挥):
- Cypher CREATE/MATCH: graphiti graphiti_core/driver/neo4j_driver.py Cypher 语法
- 连接池复用: septmuse PGVectorStore._get_cursor contextmanager 模式
- CREATE EXTENSION: PGVectorStore._create_tables 模式

差异 (简化, 非照搬 graphiti):
- 同步 (非 async, 对齐 SeptMuse 全栈)
- 单图 (graph_name 固定 "septmuse_graph", 非 graphiti 多 entity 类型)
- 边用 source_id/target_id (对齐 SQLiteGraphStore, 非图论 vertex/edge 模型)
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.graph_stores.base import GraphEdge, GraphStore

logger = get_logger(__name__)

# psycopg3 优先, psycopg2 回退 (对齐 PGVectorStore)
try:
    from psycopg_pool import ConnectionPool

    _PSYCOPG_VERSION = 3
except ImportError:
    try:
        from psycopg2.pool import ThreadedConnectionPool as ConnectionPool

        _PSYCOPG_VERSION = 2
    except ImportError as _e:
        raise ImportError(
            "Neither 'psycopg' nor 'psycopg2' is available. Install with: pip install septmuse[postgres]"
        ) from _e


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AGEGraphStore(GraphStore):
    """Apache AGE 图存储后端 (实现 GraphStore ABC)。

    零配置不可用 — 需 Postgres + Apache AGE 扩展。
    用法:
        graph = AGEGraphStore(connection_string="postgresql://user:pass@host:5432/db")
        graph.add_edge("mem-1", "mem-2", "related_to", 0.85)
        edges = graph.get_edges("mem-1")
    """

    def __init__(
        self,
        *,
        graph_name: str = "septmuse_graph",
        connection_string: str | None = None,
        connection_pool: Any | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        minconn: int = 1,
        maxconn: int = 5,
    ) -> None:
        self.graph_name = graph_name
        self._graph_ensured = False

        # 连接池: 外部传入优先, 否则自建
        if connection_pool is not None:
            self.connection_pool: Any = connection_pool
        else:
            conn_str = connection_string or f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            if _PSYCOPG_VERSION == 3:
                self.connection_pool = ConnectionPool(conninfo=conn_str, min_size=minconn, max_size=maxconn, open=False)
                self.connection_pool.open(wait=False)
            else:
                self.connection_pool = ConnectionPool(minconn=minconn, maxconn=maxconn, dsn=conn_str)

        logger.info("age_graph_store_ready", graph=graph_name)

    def _ensure_graph(self) -> None:
        """首次操作时 lazy 建图 (CREATE EXTENSION + CREATE GRAPH)。"""
        if self._graph_ensured:
            return
        with self._get_cursor(commit=True) as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            # 加载 age 扩展到当前 session
            cur.execute("SET search_path = ag_catalog, public")
            # 创建图 (SELECT ag_catalog.create_graph 如果不存在)
            cur.execute(
                "SELECT ag_catalog.create_graph(%s) WHERE NOT EXISTS "
                "(SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s)",
                (self.graph_name, self.graph_name),
            )
        self._graph_ensured = True

    def _get_cursor(self, commit: bool = False):  # type: ignore[no-untyped-def]
        """统一 cursor contextmanager (对齐 PGVectorStore._get_cursor)。"""
        if _PSYCOPG_VERSION == 3:

            @contextlib.contextmanager
            def _cursor() -> Any:
                with self.connection_pool.connection() as conn, conn.cursor() as cur:
                    try:
                        yield cur
                        if commit:
                            conn.commit()
                    except Exception:
                        conn.rollback()
                        logger.error("age_cursor_error", exc_info=True)
                        raise

            return _cursor()
        else:

            @contextlib.contextmanager
            def _cursor() -> Any:
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

            return _cursor()

    def _cypher(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        """执行 Cypher 查询 (SELECT * FROM ag_catalog.cypher(graph, $$ query $$))。

        AGE 的 Cypher 通过 ag_catalog.cypher() 函数执行, 返回 record 集合。
        $$ ... $$ 是 dollar-quoted string, 避免内部引号转义。
        """
        self._ensure_graph()
        full_query = f"SELECT * FROM ag_catalog.cypher('{self.graph_name}', $${query}$$) AS (result agtype)"
        with self._get_cursor() as cur:
            cur.execute(full_query, params)
            return cur.fetchall()

    def _cypher_write(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        """执行写 Cypher (CREATE/MERGE), commit。"""
        self._ensure_graph()
        full_query = f"SELECT * FROM ag_catalog.cypher('{self.graph_name}', $${query}$$) AS (result agtype)"
        with self._get_cursor(commit=True) as cur:
            cur.execute(full_query, params)
            return cur.fetchall()

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        score: float = 0.0,
    ) -> str:
        """添加有向边 (MERGE 幂等)。返回 edge_id。

        Cypher:
            MERGE (s:Memory {id: $src})-[:related_to {score: 0.85, edge_id: 'link-xxx'}]->(t:Memory {id: $tgt})
        """
        edge_id = f"link-{uuid.uuid4()}"
        now = _utcnow_iso()
        cypher = (
            f"MERGE (s:Memory {{id: '{source_id}'}}) "
            f"MERGE (t:Memory {{id: '{target_id}'}}) "
            f"MERGE (s)-[r:{relation} {{edge_id: '{edge_id}'}}]->(t) "
            f"SET r.score = {score}, r.created_at = '{now}'"
        )
        self._cypher_write(cypher)
        return edge_id

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """获取出边 (MATCH (s)-[r]->(t) WHERE s.id = node_id)。"""
        cypher = f"MATCH (s:Memory {{id: '{node_id}'}})-[r]->(t:Memory) RETURN r.edge_id, s.id, t.id, type(r), r.score"
        rows = self._cypher(cypher)
        return [
            GraphEdge(
                id=str(r[0]),
                source_id=str(r[1]),
                target_id=str(r[2]),
                relation=str(r[3]),
                score=float(r[4]) if r[4] is not None else 0.0,
            )
            for r in rows
        ]

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        """获取邻居节点 ID 列表。"""
        if relation is None:
            cypher = f"MATCH (s:Memory {{id: '{node_id}'}})-[:]->(t:Memory) RETURN t.id"
        else:
            cypher = f"MATCH (s:Memory {{id: '{node_id}'}})-[:{relation}]->(t:Memory) RETURN t.id"
        rows = self._cypher(cypher)
        return [str(r[0]) for r in rows]

    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        """检查边是否存在。"""
        cypher = (
            f"MATCH (s:Memory {{id: '{source_id}'}})-[:{relation}]->(t:Memory {{id: '{target_id}'}}) "
            f"RETURN count(*) > 0"
        )
        rows = self._cypher(cypher)
        return len(rows) > 0

    def delete_edge(self, edge_id: str) -> bool:
        """删除边 (Cypher)。True=删除成功, False=不存在。

        AGE Cypher DELETE 不返回 rowcount, 先 MATCH count 再 DELETE
        (对齐 has_edge 的 count 检查模式, 避免无返回值时无法判定)。
        """
        check = f"MATCH ()-[r]->() WHERE r.edge_id = '{edge_id}' RETURN count(r)"
        rows = self._cypher(check)
        exists = len(rows) > 0 and int(rows[0][0]) > 0
        if not exists:
            return False
        self._cypher_write(f"MATCH ()-[r]->() WHERE r.edge_id = '{edge_id}' DELETE r")
        return True

    def close(self) -> None:
        """关闭连接池。"""
        try:
            if _PSYCOPG_VERSION == 3:
                self.connection_pool.close()
            else:
                self.connection_pool.closeall()
        except Exception:
            pass

    # ── 节点管理 (新增) ──

    def add_node(self, node_id: str, properties: dict[str, Any] | None = None) -> None:
        props_str = ", ".join(f"n.{k} = '{v}'" for k, v in (properties or {}).items())
        set_clause = f" SET {props_str}" if props_str else ""
        self._cypher_write(f"MERGE (n:Memory {{id: '{node_id}'}}){set_clause}")

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._cypher(f"MATCH (n:Memory {{id: '{node_id}'}}) RETURN n")
        if not rows:
            return None
        return {"id": node_id, "properties": {}}

    def delete_node(self, node_id: str, *, cascade: bool = True) -> bool:
        check = self._cypher(f"MATCH (n:Memory {{id: '{node_id}'}}) RETURN count(n)")
        exists = len(check) > 0 and int(check[0][0]) > 0
        if not exists:
            return False
        if cascade:
            self._cypher_write(f"MATCH (n:Memory {{id: '{node_id}'}}) DETACH DELETE n")
        else:
            self._cypher_write(f"MATCH (n:Memory {{id: '{node_id}'}}) DELETE n")
        return True

    # ── 入边查询 (新增) ──

    def get_in_edges(self, node_id: str) -> list[GraphEdge]:
        cypher = (
            f"MATCH (s:Memory)-[r]->(t:Memory {{id: '{node_id}'}}) "
            f"RETURN r.edge_id, s.id, t.id, type(r), r.score"
        )
        rows = self._cypher(cypher)
        return [
            GraphEdge(
                id=str(r[0]),
                source_id=str(r[1]),
                target_id=str(r[2]),
                relation=str(r[3]),
                score=float(r[4]) if r[4] is not None else 0.0,
            )
            for r in rows
        ]

    # ── 图统计 (新增) ──

    def get_stats(self) -> dict[str, Any]:
        node_rows = self._cypher("MATCH (n:Memory) RETURN count(n)")
        edge_rows = self._cypher("MATCH ()-[r]->() RETURN count(r)")
        node_count = int(node_rows[0][0]) if node_rows else 0
        edge_count = int(edge_rows[0][0]) if edge_rows else 0
        max_edges = node_count * (node_count - 1) if node_count > 1 else 1
        density = (edge_count / max_edges) if max_edges > 0 else 0.0
        return {"node_count": node_count, "edge_count": edge_count, "density": round(density, 4)}
