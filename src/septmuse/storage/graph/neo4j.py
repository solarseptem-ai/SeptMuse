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
"""Neo4j 图存储后端 — extras=[neo4j] 可选实现。

借鉴 graphiti graphiti_core/driver/driver.py 的 GraphDriver 模式,
简化为 SeptMuse GraphStore 6 方法 (add_edge/get_edges/get_neighbors/has_edge/delete_edge/close)。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.graph.base import GraphEdge, GraphStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jGraphStore(GraphStore):
    """Neo4j 图存储 (extras=[neo4j])。"""

    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver: Any = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connectivity()
        self._create_constraints()
        logger.info("neo4j_graph_store_ready", uri=uri)

    def _verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}") from e

    def _create_constraints(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT memory_link_id IF NOT EXISTS FOR (e:MemoryLink) REQUIRE e.id IS UNIQUE")

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        score: float = 0.0,
    ) -> str:
        edge_id = f"edge-{uuid.uuid4()}"
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:Memory {id: $source_id})
                MERGE (m:Memory {id: $target_id})
                CREATE (n)-[e:MemoryLink {id: $edge_id, relation: $relation, score: $score, created_at: $ts}]->(m)
                """,
                source_id=source_id,
                target_id=target_id,
                edge_id=edge_id,
                relation=relation,
                score=score,
                ts=_utcnow_iso(),
            )
        return edge_id

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:Memory {id: $node_id})-[e:MemoryLink]->(m:Memory)
                RETURN e.id, e.relation, e.score, $node_id, m.id
                """,
                node_id=node_id,
            )
            return [
                GraphEdge(
                    id=record["e.id"],
                    source_id=record["$node_id"],
                    target_id=record["m.id"],
                    relation=record["e.relation"],
                    score=record["e.score"],
                )
                for record in result
            ]

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        rel_filter = "WHERE e.relation = $relation" if relation else ""
        query = f"""
            MATCH (n:Memory {{id: $node_id}})-[e:MemoryLink]->(m:Memory)
            {rel_filter}
            RETURN m.id
        """
        params = {"node_id": node_id}
        if relation:
            params["relation"] = relation
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [record["m.id"] for record in result]

    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:Memory {id: $source_id})-[e:MemoryLink {relation: $relation}]->(m:Memory {id: $target_id})
                RETURN count(e) > 0 AS exists
                """,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
            )
            return result.single()["exists"]

    def delete_edge(self, edge_id: str) -> bool:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:MemoryLink {id: $edge_id}) DELETE e RETURN count(e) AS deleted",
                edge_id=edge_id,
            )
            return result.single()["deleted"] > 0

    def close(self) -> None:
        self._driver.close()
