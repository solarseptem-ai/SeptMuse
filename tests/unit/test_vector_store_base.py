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
"""VectorStoreBase ABC 契约 + SQLiteVectorStore CRUD 测试。"""

from __future__ import annotations

import sqlite3

import pytest

from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase
from septmuse.storage.vector_stores.sqlite_vec import SQLiteVectorStore


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        VectorStoreBase()


def test_vector_search_result_dataclass():
    r = VectorSearchResult(id="x", score=0.9, payload={"k": "v"})
    assert r.id == "x"
    assert r.score == 0.9
    assert r.payload == {"k": "v"}


def test_vector_entry_dataclass():
    e = VectorEntry(id="x", vector=[0.1, 0.2], payload={"k": "v"})
    assert e.id == "x"
    assert e.vector == [0.1, 0.2]
    assert e.payload == {"k": "v"}


@pytest.fixture()
def vec_store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    store = SQLiteVectorStore(conn=conn)
    yield store
    store.close()


def test_insert_and_search(vec_store):
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ids = ["m1", "m2"]
    payloads = [{"user_id": "alice"}, {"user_id": "alice"}]
    vec_store.insert_vectors(vectors, ids, payloads)

    results = vec_store.search_vectors([1.0, 0.0, 0.0], top_k=2, filters={"user_id": "alice"})
    assert len(results) == 2
    assert results[0].id == "m1"
    assert results[0].score > 0.99  # cosine ~1.0


def test_insert_dimension_mismatch_raises(vec_store):
    with pytest.raises(ValueError, match="dimension"):
        vec_store.insert_vectors(
            [[1.0, 0.0], [1.0, 0.0, 0.0]],  # different dims
            ["m1", "m2"],
            [{"user_id": "u"}, {"user_id": "u"}],
        )


def test_delete_vector(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    assert vec_store.delete_vector("m1") is True
    assert vec_store.delete_vector("m1") is False  # 已删除


def test_get_vector(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    entry = vec_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert entry.payload == {"user_id": "u"}
    assert vec_store.get_vector("missing") is None


def test_list_vectors_with_filter(vec_store):
    vec_store.insert_vectors(
        [[1.0, 0.0], [0.0, 1.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    alice = vec_store.list_vectors(filters={"user_id": "alice"})
    assert len(alice) == 1
    assert alice[0].id == "m1"


def test_search_filters_by_payload(vec_store):
    vec_store.insert_vectors(
        [[1.0, 0.0], [1.0, 0.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    results = vec_store.search_vectors([1.0, 0.0], top_k=10, filters={"user_id": "bob"})
    assert len(results) == 1
    assert results[0].id == "m2"


def test_search_empty_store_returns_empty(vec_store):
    results = vec_store.search_vectors([1.0, 0.0], top_k=5)
    assert results == []


def test_search_zero_vector_returns_empty(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    results = vec_store.search_vectors([0.0, 0.0], top_k=5)
    assert results == []
