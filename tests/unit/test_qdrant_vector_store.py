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
"""QdrantVectorStore 测试 (integration, 默认 skip)。

需要安装 qdrant-client 且运行 Qdrant 服务: ``pip install qdrant-client``。
默认 skip (qdrant_client 未安装时 pytest.importorskip 跳过)。
运行时需真实 Qdrant 实例 (localhost:6333)。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def qdrant_store() -> Iterator:
    pytest.importorskip("qdrant_client")
    from septmuse.storage.vector.qdrant import QdrantVectorStore

    store = QdrantVectorStore(host="localhost", port=6333)
    yield store
    store.close()


def test_qdrant_insert_and_search(qdrant_store) -> None:
    qdrant_store.insert_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "alice"}],
    )
    results = qdrant_store.search_vectors([1.0, 0.0, 0.0], top_k=2, filters={"user_id": "alice"})
    assert len(results) == 2
    assert results[0].id == "m1"


def test_qdrant_delete(qdrant_store) -> None:
    qdrant_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    assert qdrant_store.delete_vector("m1") is True
    assert qdrant_store.delete_vector("m1") is False


def test_qdrant_get_vector(qdrant_store) -> None:
    qdrant_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    entry = qdrant_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"


def test_qdrant_list_vectors_filter(qdrant_store) -> None:
    qdrant_store.insert_vectors(
        [[1.0, 0.0], [0.0, 1.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    alice = qdrant_store.list_vectors(filters={"user_id": "alice"})
    assert len(alice) == 1
