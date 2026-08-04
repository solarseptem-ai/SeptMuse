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
"""PgvectorVectorStore 测试 — 降级路径（无 pgvector 时用 SQLAlchemyVectorStore）。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore


@pytest.fixture
def store():
    """用 SQLite engine 测试降级路径 (pgvector 不可用 → 回退到 SQLAlchemyVectorStore)。"""
    engine = create_engine("sqlite://", echo=False)
    s = PgvectorVectorStore(engine)
    yield s
    s.close()


def test_pgvector_fallback_insert_and_search(store):
    """降级模式: 插入+检索 (和 SQLAlchemyVectorStore 行为一致)。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"


def test_pgvector_fallback_delete(store):
    """降级模式: 删除。"""
    store.insert_vectors([[1.0]], ["m1"])
    assert store.delete_vector("m1") is True


def test_pgvector_fallback_get(store):
    """降级模式: 取单条。"""
    store.insert_vectors([[1.0, 0.5]], ["m1"], [{"topic": "test"}])
    entry = store.get_vector("m1")
    assert entry is not None
    assert entry.vector == [1.0, 0.5]
