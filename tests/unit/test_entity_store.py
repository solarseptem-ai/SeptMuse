"""EntityStore 测试 (借鉴 mem0 _upsert_entity / _remove_memory_from_entity_store)。"""

import json
import sqlite3
import threading

from septmuse.embedders.hash import HashEmbedder
from septmuse.extraction.entity import Entity
from septmuse.storage.entity_store import EntityStore


def make_store(tmp_path, embedder=None):
    """创建测试用 EntityStore (独立 SQLite, 不依赖 SQLiteMemoryStore)。"""
    db_path = str(tmp_path / "test_entities.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    lock = threading.Lock()
    return EntityStore(conn, lock, embedder=embedder)


class TestEntityStoreUpsert:
    def test_create_new_entity(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        assert eid is not None
        result = store.get(eid)
        assert result["entity_text"] == "Google"
        assert result["entity_type"] == "PROPER"
        assert "mem-001" in json.loads(result["linked_memory_ids"])

    def test_exact_match_appends_memory_id(self, tmp_path):
        store = make_store(tmp_path)
        entity1 = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        entity2 = Entity(text="google", entity_type="PROPER", start=0, end=6)
        eid1 = store.upsert(entity1, "mem-001", user_id="u1")
        eid2 = store.upsert(entity2, "mem-002", user_id="u1")
        assert eid1 == eid2
        result = store.get(eid1)
        linked = json.loads(result["linked_memory_ids"])
        assert "mem-001" in linked
        assert "mem-002" in linked

    def test_different_users_separate(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid1 = store.upsert(entity, "mem-001", user_id="u1")
        eid2 = store.upsert(entity, "mem-002", user_id="u2")
        assert eid1 != eid2

    def test_semantic_match_with_embedder(self, tmp_path):
        store = make_store(tmp_path, embedder=HashEmbedder())
        entity1 = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        entity2 = Entity(text="Google Inc", entity_type="PROPER", start=0, end=10)
        eid1 = store.upsert(entity1, "mem-001", user_id="u1")
        eid2 = store.upsert(entity2, "mem-002", user_id="u1")
        assert eid1 is not None
        assert eid2 is not None

    def test_get_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get("nonexistent-id") is None


class TestEntityStoreSearch:
    def test_search_exact_match(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Google", user_id="u1", top_k=5)
        assert any(r["entity_text"] == "Google" for r in results)

    def test_search_no_match(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Microsoft", user_id="u1", top_k=5)
        assert len(results) == 0

    def test_search_user_isolation(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Google", user_id="u2", top_k=5)
        assert len(results) == 0


class TestEntityStoreList:
    def test_list_all(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        store.upsert(Entity(text="Python", entity_type="IDENTIFIER", start=0, end=6), "m2", user_id="u1")
        result = store.list(user_id="u1")
        assert len(result) == 2

    def test_list_by_type(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        store.upsert(Entity(text="Python", entity_type="IDENTIFIER", start=0, end=6), "m2", user_id="u1")
        result = store.list(user_id="u1", entity_type="PROPER")
        assert len(result) == 1
        assert result[0]["entity_text"] == "Google"

    def test_list_user_isolation(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        result = store.list(user_id="u2")
        assert len(result) == 0


class TestEntityStoreGetLinked:
    def test_get_linked_memories(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        linked = store.get_linked_memories(eid)
        assert "mem-001" in linked

    def test_get_linked_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_linked_memories("nonexistent") == []


class TestEntityStoreRemove:
    def test_remove_memory_from_entities(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        store.remove_memory_from_entities("mem-001")
        result = store.get(eid)
        assert result is None

    def test_remove_keeps_entity_if_other_links(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        store.upsert(Entity(text="google", entity_type="PROPER", start=0, end=6), "mem-002", user_id="u1")
        store.remove_memory_from_entities("mem-001")
        result = store.get(eid)
        assert result is not None
        linked = json.loads(result["linked_memory_ids"])
        assert "mem-001" not in linked
        assert "mem-002" in linked

    def test_remove_nonexistent_memory(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        store.remove_memory_from_entities("nonexistent-mem")
        result = store.get(eid)
        assert result is not None
