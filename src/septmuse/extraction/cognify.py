"""cognify 知识图谱构建流水线 (借鉴 cognee cognify + graphiti extract_nodes_and_edges)。

Pipeline: text → extract_triplets → upsert_entities → store_relations → link_memories

复用已有组件:
- TripletExtractor (P0-Task 2): 单次 LLM 联合抽取实体+边
- EntityStore (P0-Task 4): 实体向量库 (upsert + 语义去重)
- ZettelLinker (阶段3): 记忆间向量相似度双向链接
- GraphStore (阶段3): memory_links 表 (记忆间边)

新增:
- entity_relations 表: 实体间关系边 (source_entity, relation, target_entity, user_id)

SeptMuse 流程:
1. 存 verbatim memory (store.add) → memory_id
2. TripletExtractor.extract(text) → list[Triplet]
3. 每个 subject/object → EntityStore.upsert (实体去重 + linked_memory_ids)
4. 每个 (subject, predicate, object) → entity_relations 表
5. ZettelLinker.link_on_add (记忆间向量链接)
6. 返回 summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.extraction.entity import Entity, EntityExtractor
from septmuse.extraction.triplet import Triplet, TripletExtractor
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.entity_store import EntityStore
from septmuse.storage.graph.base import GraphStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CognifyPipeline:
    """cognify 知识图谱构建流水线 (借鉴 cognee cognify + graphiti)。

    依赖注入所有组件, 便于测试。
    """

    def __init__(
        self,
        store: MemoryStore,
        graph_store: GraphStore | None,
        embedder: Embedder,
        entity_store: EntityStore | None = None,
        llm: LLM | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.store = store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_store = entity_store
        self.triplet_extractor = TripletExtractor(llm=llm, entity_extractor=entity_extractor)

        if entity_store is not None:
            self._conn = entity_store._conn
            self._lock = entity_store._lock
            self._init_relations_table()

    def _init_relations_table(self) -> None:
        """建 entity_relations 表 (存实体间关系边)。"""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_relations (
                    id            TEXT PRIMARY KEY,
                    source_entity TEXT NOT NULL,
                    relation      TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    user_id       TEXT NOT NULL,
                    memory_id     TEXT,
                    created_at    TEXT NOT NULL,
                    UNIQUE(source_entity, relation, target_entity, user_id)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_entity, user_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_er_target ON entity_relations(target_entity, user_id)")
            self._conn.commit()

    def cognify(
        self,
        text: str,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """构建知识图谱: 存记忆 → 抽三元组 → 存实体/关系 → 建记忆链接。

        Returns:
            {"memory_id", "triplets", "entities", "relations", "links"}
        """
        emb = self.embedder.embed(text)
        memory_id = self.store.add(text, emb, user_id=user_id, agent_id=agent_id)

        triplets = self.triplet_extractor.extract(text)
        logger.info("cognify_triplets", memory_id=memory_id, count=len(triplets))

        entities_upserted: list[str] = []
        relations_stored: list[dict[str, str]] = []

        for triplet in triplets:
            if self.entity_store is not None:
                for entity_text in (triplet.subject, triplet.object):
                    entity = Entity(text=entity_text, entity_type="PROPER", start=0, end=len(entity_text))
                    eid = self.entity_store.upsert(entity, memory_id, user_id=user_id, agent_id=agent_id)
                    if eid not in entities_upserted:
                        entities_upserted.append(eid)

            if self.entity_store is not None:
                self._store_relation(triplet, user_id=user_id, memory_id=memory_id)
                relations_stored.append(
                    {"source": triplet.subject, "relation": triplet.predicate, "target": triplet.object}
                )

        links: list[dict[str, Any]] = []
        if self.graph_store is not None:
            from septmuse.evolution.zettel import ZettelLinker

            linker = ZettelLinker(self.store, self.graph_store, self.embedder)
            created = linker.link_on_add(memory_id, text, emb, user_id=user_id)
            links = [{"id": link.id, "target_id": link.target_id, "score": link.score} for link in created]

        logger.info(
            "cognify_done",
            memory_id=memory_id,
            triplets=len(triplets),
            entities=len(entities_upserted),
            relations=len(relations_stored),
            links=len(links),
        )

        return {
            "memory_id": memory_id,
            "triplets": [t.as_tuple() for t in triplets],
            "entities": entities_upserted,
            "relations": relations_stored,
            "links": links,
        }

    def _store_relation(self, triplet: Triplet, *, user_id: str, memory_id: str) -> None:
        """存实体间关系到 entity_relations 表 (幂等)。"""
        rel_id = str(uuid.uuid4())
        now = _utcnow_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO entity_relations
                    (id, source_entity, relation, target_entity, user_id, memory_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rel_id, triplet.subject, triplet.predicate, triplet.object, user_id, memory_id, now),
            )
            self._conn.commit()

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体 (委托 EntityStore.search)。"""
        if self.entity_store is None:
            return []
        return self.entity_store.search(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_name: str, *, user_id: str) -> list[dict[str, Any]]:
        """获取实体的邻居 (查 entity_relations 表, 双向)。

        返回 [{"entity", "relation", "direction"}]。
        direction: "outgoing" (entity 是 source) / "incoming" (entity 是 target)。
        """
        if self.entity_store is None:
            return []

        neighbors: list[dict[str, Any]] = []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT target_entity, relation FROM entity_relations
                WHERE source_entity=? AND user_id=?
                """,
                (entity_name, user_id),
            )
            for r in cur.fetchall():
                neighbors.append({"entity": r[0], "relation": r[1], "direction": "outgoing"})

            cur = self._conn.execute(
                """
                SELECT source_entity, relation FROM entity_relations
                WHERE target_entity=? AND user_id=?
                """,
                (entity_name, user_id),
            )
            for r in cur.fetchall():
                neighbors.append({"entity": r[0], "relation": r[1], "direction": "incoming"})

        return neighbors
