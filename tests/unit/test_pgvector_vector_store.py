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

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores import pgvector_store as pgvs
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


def test_pgvector_init_creates_hnsw_index(monkeypatch):
    """_init_pgvector 执行 HNSW 索引创建 SQL (不依赖真实 pgvector/Postgres)。

    验证:
    - CREATE EXTENSION IF NOT EXISTS vector
    - CREATE TABLE IF NOT EXISTS vector_entries
    - CREATE INDEX IF NOT EXISTS ... USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64)
    """
    # 强制 PGVECTOR_AVAILABLE = True (模拟 pgvector 已安装)
    monkeypatch.setattr(pgvs, "PGVECTOR_AVAILABLE", True)

    # Mock engine: 捕获执行的 SQL
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.dialect.name = "postgresql"

    PgvectorVectorStore(mock_engine, vector_dim=512)

    # 提取所有执行的 SQL
    sql_calls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]

    # 验证 HNSW 索引创建 SQL
    hnsw_sql = [s for s in sql_calls if "hnsw" in s.lower()]
    assert len(hnsw_sql) == 1, f"HNSW 索引创建 SQL 未找到, 实际执行: {sql_calls}"
    assert "vector_cosine_ops" in hnsw_sql[0]
    assert "m = 16" in hnsw_sql[0]
    assert "ef_construction = 64" in hnsw_sql[0]

    # 验证 CREATE EXTENSION 和 CREATE TABLE 也执行了
    assert any("CREATE EXTENSION" in s for s in sql_calls)
    assert any("CREATE TABLE" in s for s in sql_calls)
