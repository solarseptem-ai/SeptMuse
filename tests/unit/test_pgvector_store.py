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
"""PGVectorStore 单元测试。

单元测试: mock ConnectionPool, 不需要真实 Postgres。
集成测试: @pytest.mark.skipif 需要 SEPTMUSE_TEST_PG_DSN 环境变量 + 真实 Postgres。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from septmuse.storage.base import MemoryStore

# PGVectorStore 需要 psycopg2 (当前环境已安装 psycopg2 2.9.11)
# psycopg3 未安装, 走 psycopg2 回退路径
from septmuse.storage.vector.pgvector import (
    PGVectorStore,
    _to_pgvector_str,
    _with_sslmode,
)

# 集成测试 gate: 需要环境变量 SEPTMUSE_TEST_PG_DSN 指向真实 Postgres
HAS_PG_DSN = bool(os.getenv("SEPTMUSE_TEST_PG_DSN"))


# ======================================================================
# 辅助函数测试 (不需要 Postgres)
# ======================================================================


class TestToPgvectorStr:
    def test_basic(self) -> None:
        result = _to_pgvector_str([0.1, 0.2, 0.3])
        assert result == "[0.1,0.2,0.3]"

    def test_single_element(self) -> None:
        result = _to_pgvector_str([1.0])
        assert result == "[1.0]"

    def test_empty_list(self) -> None:
        result = _to_pgvector_str([])
        assert result == "[]"

    def test_negative_values(self) -> None:
        result = _to_pgvector_str([-0.5, 0.5])
        assert "-0.5" in result
        assert "0.5" in result


class TestWithSslmode:
    def test_uri_format_adds_sslmode(self) -> None:
        result = _with_sslmode("postgresql://user:pass@host:5432/db", "require")
        assert "sslmode=require" in result

    def test_conninfo_format_adds_sslmode(self) -> None:
        result = _with_sslmode("host=localhost dbname=test", "require")
        assert "sslmode=require" in result

    def test_uri_replaces_existing_sslmode(self) -> None:
        result = _with_sslmode("postgresql://user:pass@host:5432/db?sslmode=disable", "require")
        assert "sslmode=require" in result
        assert "sslmode=disable" not in result

    def test_conninfo_replaces_existing_sslmode(self) -> None:
        result = _with_sslmode("host=localhost dbname=test sslmode=disable", "require")
        assert "sslmode=require" in result
        assert "sslmode=disable" not in result


# ======================================================================
# PGVectorStore 构造测试 (mock ConnectionPool)
# ======================================================================


@pytest.fixture()
def mock_pool(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock pgvector 模块中的 ConnectionPool, 避免真实 Postgres 连接。

    pgvector.py import 时把 ConnectionPool 绑定为 psycopg2.pool.ThreadedConnectionPool,
    所以必须 mock 模块内的名称, 而非原始类。
    """
    mock = MagicMock()
    monkeypatch.setattr("septmuse.storage.vector.pgvector.ConnectionPool", MagicMock(return_value=mock))
    return mock


class TestPGVectorStoreInit:
    def test_inherits_memory_store(self, mock_pool: MagicMock) -> None:
        assert isinstance(PGVectorStore(connection_string="postgresql://t:t@h:5432/d"), MemoryStore)

    def test_default_dims(self, mock_pool: MagicMock) -> None:
        store = PGVectorStore(connection_string="postgresql://t:t@h:5432/d")
        assert store._dim == 1536

    def test_custom_dims(self, mock_pool: MagicMock) -> None:
        store = PGVectorStore(connection_string="postgresql://t:t@h:5432/d", embedding_model_dims=768)
        assert store._dim == 768

    def test_connection_string_priority(self, mock_pool: MagicMock) -> None:
        store = PGVectorStore(
            connection_string="postgresql://custom:custom@customhost:5432/customdb",
            dbname="ignored",
            user="ignored",
            password="ignored",
        )
        assert store.collection_name == "memories"

    def test_sslmode_injection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_constructor = MagicMock()
        monkeypatch.setattr("septmuse.storage.vector.pgvector.ConnectionPool", mock_constructor)
        PGVectorStore(
            connection_string="postgresql://t:t@h:5432/d",
            sslmode="require",
        )
        call_kwargs = mock_constructor.call_args.kwargs
        dsn = call_kwargs.get("dsn") or ""
        assert "sslmode=require" in dsn

    def test_collection_not_ensured_until_operation(self, mock_pool: MagicMock) -> None:
        store = PGVectorStore(connection_string="postgresql://t:t@h:5432/d")
        assert store._collection_ensured is False


# ======================================================================
# 集成测试 (需要真实 Postgres + SEPTMUSE_TEST_PG_DSN)
# ======================================================================


@pytest.mark.skipif(not HAS_PG_DSN, reason="Set SEPTMUSE_TEST_PG_DSN to run Postgres integration tests")
class TestPGVectorStoreIntegration:
    """集成测试: 需要真实 Postgres + pgvector 扩展。

    设置环境变量运行:
        $env:SEPTMUSE_TEST_PG_DSN = "postgresql://user:pass@host:5432/dbname"
        pytest tests/unit/test_pgvector_store.py -k Integration
    """

    @pytest.fixture()
    def store(self) -> Iterator[PGVectorStore]:
        dsn = os.getenv("SEPTMUSE_TEST_PG_DSN", "")
        s = PGVectorStore(connection_string=dsn, embedding_model_dims=3)
        yield s
        s.close()

    def test_add_and_search(self, store: PGVectorStore) -> None:
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        store.add("doc1", emb1, user_id="alice")
        store.add("doc2", emb2, user_id="alice")

        results = store.search([0.9, 0.1, 0.0], user_id="alice", top_k=2)
        assert len(results) >= 1
        assert results[0]["memory"] == "doc1"
        assert results[0]["score"] > 0.5

    def test_search_threshold(self, store: PGVectorStore) -> None:
        store.add("similar", [1.0, 0.0, 0.0], user_id="bob")
        store.add("different", [0.0, 0.0, 1.0], user_id="bob")

        results = store.search([1.0, 0.0, 0.0], user_id="bob", top_k=5, threshold=0.9)
        assert all(r["score"] >= 0.9 for r in results)
        assert len(results) <= 2

    def test_get_all_by_user(self, store: PGVectorStore) -> None:
        store.add("doc-a", [1.0, 0.0, 0.0], user_id="carol")
        store.add("doc-b", [0.0, 1.0, 0.0], user_id="carol")
        store.add("doc-c", [0.0, 0.0, 1.0], user_id="dave")

        carol_memories = store.get_all(user_id="carol")
        assert len(carol_memories) == 2
        dave_memories = store.get_all(user_id="dave")
        assert len(dave_memories) == 1

    def test_get_returns_none_for_missing(self, store: PGVectorStore) -> None:
        assert store.get("nonexistent-id") is None

    def test_delete_is_soft(self, store: PGVectorStore) -> None:
        mid = store.add("to-delete", [1.0, 0.0, 0.0], user_id="eve")
        store.delete(mid)
        # get 返回 None (is_deleted=0 过滤)
        assert store.get(mid) is None
        # get_all 不返回
        assert all(m["id"] != mid for m in store.get_all(user_id="eve"))
