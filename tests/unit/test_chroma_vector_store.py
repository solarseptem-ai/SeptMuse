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
"""ChromaVectorStore 测试 (integration, 默认 skip)。

需要安装 chromadb: ``pip install chromadb``。
默认 skip (chromadb 未安装时 pytest.importorskip 跳过)。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def chroma_store(tmp_path: Path) -> Iterator:
    pytest.importorskip("chromadb")
    from septmuse.storage.vector_stores.chroma import ChromaVectorStore

    store = ChromaVectorStore(persist_path=str(tmp_path / "chroma"))
    yield store
    store.close()


def test_chroma_insert_and_search(chroma_store) -> None:
    chroma_store.insert_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "alice"}],
    )
    results = chroma_store.search_vectors([1.0, 0.0, 0.0], top_k=2, filters={"user_id": "alice"})
    assert len(results) == 2
    assert results[0].id == "m1"


def test_chroma_delete(chroma_store) -> None:
    chroma_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    assert chroma_store.delete_vector("m1") is True
    assert chroma_store.delete_vector("m1") is False


def test_chroma_get_vector(chroma_store) -> None:
    chroma_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    entry = chroma_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"


def test_chroma_list_vectors_filter(chroma_store) -> None:
    chroma_store.insert_vectors(
        [[1.0, 0.0], [0.0, 1.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    alice = chroma_store.list_vectors(filters={"user_id": "alice"})
    assert len(alice) == 1
