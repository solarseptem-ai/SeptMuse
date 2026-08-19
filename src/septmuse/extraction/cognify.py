"""cognify 知识图谱构建流水线。"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.extraction.entity import Entity, EntityExtractor
from septmuse.extraction.triplet import Triplet, TripletExtractor
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore
from septmuse.storage.relational_stores.entity_store import EntityStore

logger = get_logger(__name__)


class CognifyPipeline:
    """cognify 知识图谱构建流水线。"""

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

    def cognify(
        self,
        text: str,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """构建知识图谱。"""
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
        """存实体间关系 (委托 EntityStore.store_relation, 幂等)。"""
        if self.entity_store is None:
            return
        self.entity_store.store_relation(
            triplet.subject, triplet.predicate, triplet.object,
            user_id=user_id, memory_id=memory_id,
        )

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体 (委托 EntityStore.search)。"""
        if self.entity_store is None:
            return []
        return self.entity_store.search(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_name: str, *, user_id: str) -> list[dict[str, Any]]:
        """获取实体的邻居 (委托 EntityStore.get_neighbors)。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_neighbors(entity_name, user_id=user_id)
