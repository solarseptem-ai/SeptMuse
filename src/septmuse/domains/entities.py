#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""EntityModule — 实体领域模块（search_entities / get_entity_neighbors / list_entities / get_entity_relations / list_agents / is_cross_agent）。"""

from __future__ import annotations

from typing import Any

from septmuse.storage.base import MemoryStore, SharedQuery


class EntityModule:
    """实体领域模块 — 实体 CRUD + 邻居遍历 + 跨 agent 共享。"""

    def __init__(
        self,
        store: MemoryStore,
        entity_store: Any | None = None,
        entity_extractor: Any | None = None,
    ) -> None:
        self.store = store
        self.entity_store = entity_store
        self.entity_extractor = entity_extractor

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """抽取实体（不存储）。"""
        if self.entity_extractor is None:
            return []
        entities = self.entity_extractor.extract(text)
        return [{"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end} for e in entities]

    def add_entity(
        self, entity_text: str, entity_type: str, memory_id: str, *, user_id: str
    ) -> dict[str, Any]:
        """手动添加实体。"""
        if self.entity_store is None:
            return {"event": "SKIP", "entity": entity_text, "reason": "no entity_store"}
        from septmuse.extraction.entity import Entity

        entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=len(entity_text))
        eid = self.entity_store.upsert(entity, memory_id, user_id=user_id)
        return {"event": "ADD", "entity": entity_text, "entity_id": eid}

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体。"""
        if self.entity_store is None:
            return []
        return self.entity_store.search(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_id: str) -> list[dict[str, Any]]:
        """获取实体的邻居记忆。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_neighbors(entity_id)

    def list_entities(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出实体。"""
        if self.entity_store is None:
            return []
        return self.entity_store.list(user_id=user_id, entity_type=entity_type, limit=limit)

    def get_entity_relations(
        self, entity_name: str, *, user_id: str
    ) -> list[dict[str, Any]]:
        """实体间关系遍历。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_neighbors(entity_name, user_id=user_id)

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent。"""
        if not isinstance(self.store, SharedQuery):
            return []
        from septmuse.governance.sharing import SharedMemoryAccessor

        return SharedMemoryAccessor(self.store).list_agents(user_id)

    def is_cross_agent(self, user_id: str) -> bool:
        """检查是否跨 agent 共享。"""
        from septmuse.governance.sharing import SharedMemoryAccessor

        return SharedMemoryAccessor(self.store).is_cross_agent(user_id)
