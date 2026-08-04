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
"""GraphStore 单元测试。

SQLiteGraphStore: 用真实 SQLite 内存数据库测试 (零外部依赖)。
AGEGraphStore: mock ConnectionPool 单元测试 + skipif 集成测试。
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from sqlmodel import create_engine

from septmuse.storage.graph_stores.base import GraphEdge, GraphStore
from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

# ======================================================================
# SQLiteGraphStore 单元测试 (真实 SQLite, 零外部依赖)
# ======================================================================


@pytest.fixture()
def graph_store() -> Iterator[SQLiteGraphStore]:
    """用 :memory: SQLite 构造 SQLiteGraphStore (复用 ORMMemoryStore 的 engine)。"""
    engine = create_engine("sqlite://")
    store = ORMMemoryStore(engine)
    raw_conn = store.engine.raw_connection()
    graph = SQLiteGraphStore(raw_conn, threading.Lock())
    yield graph
    store.close()


class TestSQLiteGraphStoreInheritance:
    def test_inherits_graph_store(self, graph_store: SQLiteGraphStore) -> None:
        assert isinstance(graph_store, GraphStore)


class TestSQLiteGraphStoreAddEdge:
    def test_add_edge_returns_id(self, graph_store: SQLiteGraphStore) -> None:
        edge_id = graph_store.add_edge("mem-1", "mem-2", "related_to", 0.85)
        assert edge_id.startswith("link-")

    def test_add_edge_idempotent(self, graph_store: SQLiteGraphStore) -> None:
        """重复添加相同 source+target+relation 不报错 (UNIQUE 约束)。"""
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.85)
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.90)  # 不同 score
        edges = graph_store.get_edges("mem-1")
        # 只有 1 条边 (UNIQUE 去重)
        assert len(edges) == 1

    def test_add_edge_default_relation(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2")
        edges = graph_store.get_edges("mem-1")
        assert len(edges) == 1
        assert edges[0].relation == "related_to"
        assert edges[0].score == 0.0


class TestSQLiteGraphStoreGetEdges:
    def test_get_edges_returns_outgoing(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph_store.add_edge("mem-1", "mem-3", "causes", 0.5)
        graph_store.add_edge("mem-2", "mem-1", "related_to", 0.7)  # 反向边

        edges = graph_store.get_edges("mem-1")
        assert len(edges) == 2  # 只返回出边, 不含反向
        targets = {e.target_id for e in edges}
        assert targets == {"mem-2", "mem-3"}

    def test_get_edges_empty(self, graph_store: SQLiteGraphStore) -> None:
        assert graph_store.get_edges("nonexistent") == []

    def test_get_edges_preserves_fields(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.85)
        edges = graph_store.get_edges("mem-1")
        assert len(edges) == 1
        e = edges[0]
        assert isinstance(e, GraphEdge)
        assert e.source_id == "mem-1"
        assert e.target_id == "mem-2"
        assert e.relation == "related_to"
        assert e.score == 0.85


class TestSQLiteGraphStoreGetNeighbors:
    def test_get_neighbors_all_relations(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph_store.add_edge("mem-1", "mem-3", "causes", 0.5)
        graph_store.add_edge("mem-2", "mem-1", "related_to", 0.7)

        neighbors = graph_store.get_neighbors("mem-1")
        assert set(neighbors) == {"mem-2", "mem-3"}

    def test_get_neighbors_filtered_by_relation(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph_store.add_edge("mem-1", "mem-3", "causes", 0.5)

        neighbors = graph_store.get_neighbors("mem-1", relation="causes")
        assert neighbors == ["mem-3"]

    def test_get_neighbors_empty(self, graph_store: SQLiteGraphStore) -> None:
        assert graph_store.get_neighbors("nonexistent") == []


class TestSQLiteGraphStoreHasEdge:
    def test_has_edge_true(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        assert graph_store.has_edge("mem-1", "mem-2", "related_to")

    def test_has_edge_false_wrong_relation(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        assert not graph_store.has_edge("mem-1", "mem-2", "causes")

    def test_has_edge_false_wrong_target(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        assert not graph_store.has_edge("mem-1", "mem-3", "related_to")

    def test_has_edge_false_nonexistent(self, graph_store: SQLiteGraphStore) -> None:
        assert not graph_store.has_edge("mem-1", "mem-2", "related_to")


class TestSQLiteGraphStoreBidirectional:
    """验证双向链接 (ZettelLinker 用法: add_edge 两次创建双向)。"""

    def test_bidirectional_links(self, graph_store: SQLiteGraphStore) -> None:
        graph_store.add_edge("mem-1", "mem-2", "related_to", 0.85)
        graph_store.add_edge("mem-2", "mem-1", "related_to", 0.85)

        assert graph_store.has_edge("mem-1", "mem-2", "related_to")
        assert graph_store.has_edge("mem-2", "mem-1", "related_to")

        neighbors_1 = graph_store.get_neighbors("mem-1")
        neighbors_2 = graph_store.get_neighbors("mem-2")
        assert neighbors_1 == ["mem-2"]
        assert neighbors_2 == ["mem-1"]


# ======================================================================
# AGEGraphStore 单元测试 (mock ConnectionPool)
# ======================================================================


@pytest.fixture()
def mock_age_pool(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock AGEGraphStore 的 ConnectionPool (避免真实 Postgres)。"""
    mock = MagicMock()
    monkeypatch.setattr("septmuse.storage.graph_stores.age.ConnectionPool", MagicMock(return_value=mock))
    return mock


class TestAGEGraphStoreInit:
    def test_inherits_graph_store(self, mock_age_pool: MagicMock) -> None:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        store = AGEGraphStore(connection_string="postgresql://t:t@h:5432/d")
        assert isinstance(store, GraphStore)

    def test_default_graph_name(self, mock_age_pool: MagicMock) -> None:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        store = AGEGraphStore(connection_string="postgresql://t:t@h:5432/d")
        assert store.graph_name == "septmuse_graph"

    def test_custom_graph_name(self, mock_age_pool: MagicMock) -> None:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        store = AGEGraphStore(connection_string="postgresql://t:t@h:5432/d", graph_name="custom")
        assert store.graph_name == "custom"

    def test_connection_string_priority(self, mock_age_pool: MagicMock) -> None:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        store = AGEGraphStore(
            connection_string="postgresql://custom:custom@customhost:5432/customdb",
            dbname="ignored",
        )
        assert store.graph_name == "septmuse_graph"

    def test_external_connection_pool(self, mock_age_pool: MagicMock) -> None:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        external_pool = MagicMock()
        store = AGEGraphStore(connection_pool=external_pool)
        assert store.connection_pool is external_pool


# ======================================================================
# MemoryStore 关系查询测试 (ORMMemoryStore, 验证新方法)
# ======================================================================


@pytest.fixture()
def store_with_data() -> Iterator[ORMMemoryStore]:
    """构造有跨 agent 数据的 ORMMemoryStore。"""
    engine = create_engine("sqlite://")
    store = ORMMemoryStore(engine)
    store.add("alice doc 1", [1.0, 0.0], user_id="alice", agent_id="bot1")
    store.add("alice doc 2", [0.0, 1.0], user_id="alice", agent_id="bot2")
    store.add("alice shared", [1.0, 1.0], user_id="alice", agent_id=None)  # 跨 agent 共享
    store.add("bob doc", [0.5, 0.5], user_id="bob", agent_id="bot1")
    yield store
    store.close()


class TestMemoryStoreRelationQueries:
    def test_list_agents(self, store_with_data: ORMMemoryStore) -> None:
        agents = store_with_data.list_agents("alice")
        assert set(agents) == {"bot1", "bot2"}  # NULL agent_id 被排除

    def test_list_agents_empty(self, store_with_data: ORMMemoryStore) -> None:
        assert store_with_data.list_agents("nonexistent") == []

    def test_list_users(self, store_with_data: ORMMemoryStore) -> None:
        users = store_with_data.list_users("bot1")
        assert set(users) == {"alice", "bob"}

    def test_list_users_empty(self, store_with_data: ORMMemoryStore) -> None:
        assert store_with_data.list_users("nonexistent") == []

    def test_get_shared_memories(self, store_with_data: ORMMemoryStore) -> None:
        memories = store_with_data.get_shared_memories("alice", limit=100)
        assert len(memories) == 3
        # 验证字段完整
        for m in memories:
            assert "id" in m
            assert "user_id" in m
            assert "agent_id" in m
            assert "memory" in m
            assert "metadata" in m
            assert "created_at" in m

    def test_get_shared_memories_limit(self, store_with_data: ORMMemoryStore) -> None:
        memories = store_with_data.get_shared_memories("alice", limit=2)
        assert len(memories) == 2

    def test_get_shared_memories_empty(self, store_with_data: ORMMemoryStore) -> None:
        assert store_with_data.get_shared_memories("nonexistent") == []


# ======================================================================
# AGEGraphStore 集成测试 (需要真实 Postgres + AGE 扩展)
# ======================================================================

HAS_PG_DSN = bool(os.getenv("SEPTMUSE_TEST_PG_DSN"))


@pytest.mark.skipif(not HAS_PG_DSN, reason="Set SEPTMUSE_TEST_PG_DSN to run AGE integration tests")
class TestAGEGraphStoreIntegration:
    """集成测试: 需要真实 Postgres + Apache AGE 扩展。

    设置环境变量运行:
        $env:SEPTMUSE_TEST_PG_DSN = "postgresql://user:pass@host:5432/dbname"
        pytest tests/unit/test_graph_store.py -k AGEIntegration
    """

    @pytest.fixture()
    def age_store(self) -> Iterator:
        from septmuse.storage.graph_stores.age import AGEGraphStore

        dsn = os.getenv("SEPTMUSE_TEST_PG_DSN", "")
        store = AGEGraphStore(connection_string=dsn)
        yield store
        store.close()

    def test_add_and_get_edges(self, age_store) -> None:
        age_store.add_edge("mem-1", "mem-2", "related_to", 0.85)
        edges = age_store.get_edges("mem-1")
        assert len(edges) >= 1
        assert any(e.target_id == "mem-2" for e in edges)

    def test_has_edge(self, age_store) -> None:
        age_store.add_edge("mem-1", "mem-3", "causes", 0.5)
        assert age_store.has_edge("mem-1", "mem-3", "causes")
        assert not age_store.has_edge("mem-1", "mem-3", "related_to")

    def test_get_neighbors(self, age_store) -> None:
        age_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        age_store.add_edge("mem-1", "mem-3", "causes", 0.5)
        neighbors = age_store.get_neighbors("mem-1")
        assert "mem-2" in neighbors
        assert "mem-3" in neighbors

    def test_get_neighbors_filtered(self, age_store) -> None:
        age_store.add_edge("mem-1", "mem-2", "related_to", 0.8)
        age_store.add_edge("mem-1", "mem-3", "causes", 0.5)
        neighbors = age_store.get_neighbors("mem-1", relation="causes")
        assert "mem-3" in neighbors
