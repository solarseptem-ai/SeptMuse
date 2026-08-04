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
"""SQLAlchemyVectorStore 测试 — 通用 JSON+numpy 向量存储。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore


@pytest.fixture
def store():
    engine = create_engine("sqlite://", echo=False)
    s = SQLAlchemyVectorStore(engine)
    yield s
    s.close()


def test_insert_and_search(store):
    """插入向量后能检索到。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"
    assert results[0].score >= 0.9


def test_search_filters_by_payload(store):
    """payload 过滤生效。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    store.insert_vectors([[0.0, 1.0]], ["m2"], [{"user_id": "bob"}])
    results = store.search_vectors([1.0, 0.0], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"


def test_delete_vector(store):
    """删除向量。"""
    store.insert_vectors([[1.0]], ["m1"])
    assert store.delete_vector("m1") is True
    assert store.delete_vector("m1") is False  # 已删


def test_get_vector(store):
    """取单条向量。"""
    store.insert_vectors([[1.0, 0.5]], ["m1"], [{"topic": "test"}])
    entry = store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert entry.vector == [1.0, 0.5]
    assert entry.payload == {"topic": "test"}


def test_get_vector_not_found(store):
    """取不存在返回 None。"""
    assert store.get_vector("nonexistent") is None


def test_list_vectors(store):
    """列向量。"""
    store.insert_vectors([[1.0], [2.0]], ["m1", "m2"], [{"user_id": "a"}, {"user_id": "b"}])
    all_vecs = store.list_vectors()
    assert len(all_vecs) == 2
    filtered = store.list_vectors(filters={"user_id": "a"})
    assert len(filtered) == 1


def test_search_empty_store(store):
    """空库检索返回空列表。"""
    results = store.search_vectors([1.0], top_k=5)
    assert results == []


def test_dimension_mismatch_raises(store):
    """维度不一致报错。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"])
    with pytest.raises(ValueError, match="dimension"):
        store.search_vectors([1.0])
