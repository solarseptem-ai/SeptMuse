# Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EntityStore.from_engine ORM CRUD 全量测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import Session, create_engine

from septmuse.extraction.entity import Entity
from septmuse.services.database.models.entity import EntityTable
from septmuse.storage.relational_stores.entity_store import EntityStore


def _make_engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'entity.db'}")


def test_from_engine_creates_table(tmp_path):
    """from_engine 自动建 septmuse_entities 表。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    assert store._engine is engine
    from sqlalchemy import inspect

    assert inspect(engine).has_table("septmuse_entities")


def test_upsert_new_entity(tmp_path):
    """新建实体 → 返回 entity_id，linked_memory_ids 含 memory_id。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")
    assert eid is not None

    result = store.get(eid)
    assert result is not None
    assert result["entity_text"] == "Alice"
    assert result["entity_type"] == "PROPER"
    import json

    assert "mem-1" in json.loads(result["linked_memory_ids"])


def test_upsert_exact_match_appends(tmp_path):
    """精确归一化匹配 → linked_memory_ids 追加。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Google", entity_type="PROPER")

    eid1 = store.upsert(entity, memory_id="mem-1", user_id="u1")
    eid2 = store.upsert(entity, memory_id="mem-2", user_id="u1")
    assert eid1 == eid2  # 同一实体

    result = store.get(eid1)
    import json

    linked = json.loads(result["linked_memory_ids"])
    assert "mem-1" in linked
    assert "mem-2" in linked


def test_upsert_different_users_separate(tmp_path):
    """不同 user_id 的同名实体独立存储。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")

    eid1 = store.upsert(entity, memory_id="mem-1", user_id="u1")
    eid2 = store.upsert(entity, memory_id="mem-2", user_id="u2")
    assert eid1 != eid2


def test_search_exact_match(tmp_path):
    """search 精确匹配返回结果。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    store.upsert(entity, memory_id="mem-1", user_id="u1")

    results = store.search("Alice", user_id="u1")
    assert len(results) == 1
    assert results[0]["entity_text"] == "Alice"
    assert results[0]["score"] == 1.0


def test_list_entities(tmp_path):
    """list 返回用户全部实体。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    store.upsert(Entity(text="Alice", entity_type="PROPER"), memory_id="m1", user_id="u1")
    store.upsert(Entity(text="Google", entity_type="PROPER"), memory_id="m2", user_id="u1")

    entities = store.list(user_id="u1")
    assert len(entities) == 2


def test_list_by_type(tmp_path):
    """list 按 entity_type 过滤。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    store.upsert(Entity(text="Alice", entity_type="PROPER"), memory_id="m1", user_id="u1")
    store.upsert(Entity(text="Python", entity_type="TOPIC"), memory_id="m2", user_id="u1")

    proper = store.list(user_id="u1", entity_type="PROPER")
    assert len(proper) == 1
    assert proper[0]["entity_text"] == "Alice"


def test_get_linked_memories(tmp_path):
    """get_linked_memories 返回 linked_memory_ids 列表。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")

    linked = store.get_linked_memories(eid)
    assert "mem-1" in linked


def test_remove_memory_from_entities(tmp_path):
    """remove_memory_from_entities 清理引用 + 空时软删除。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")

    store.remove_memory_from_entities("mem-1")

    # 只有一个 memory_id，清空后软删除
    assert store.get(eid) is None

    with Session(engine) as session:
        from sqlmodel import select

        stmt = select(EntityTable).where(EntityTable.id == eid)
        row = session.exec(stmt).first()
        assert row is not None
        assert row.is_deleted == 1


def test_remove_memory_keeps_entity_with_other_links(tmp_path):
    """多个 memory_id 时只移除一个，保留实体。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")
    store.upsert(entity, memory_id="mem-2", user_id="u1")

    store.remove_memory_from_entities("mem-1")

    with Session(engine) as session:
        from sqlmodel import select

        stmt = select(EntityTable).where(EntityTable.id == eid)
        row = session.exec(stmt).first()
        assert row is not None
        assert row.is_deleted == 0
        import json

        linked = json.loads(row.linked_memory_ids)
        assert "mem-1" not in linked
        assert "mem-2" in linked
