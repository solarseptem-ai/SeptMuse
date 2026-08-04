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
"""Neo4jGraphStore 测试 (integration, 默认 skip)。"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def neo4j_store():
    if not os.getenv("SEPTMUSE_TEST_NEO4J_URI"):
        pytest.skip("Set SEPTMUSE_TEST_NEO4J_URI to run Neo4j integration tests")
    pytest.importorskip("neo4j")
    from septmuse.storage.graph_stores.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(
        uri=os.getenv("SEPTMUSE_TEST_NEO4J_URI"),
        username=os.getenv("SEPTMUSE_TEST_NEO4J_USER", "neo4j"),
        password=os.getenv("SEPTMUSE_TEST_NEO4J_PASSWORD", ""),
    )
    yield store
    store.close()


def test_neo4j_add_and_get_edges(neo4j_store):
    edge_id = neo4j_store.add_edge("m1", "m2", "related_to", 0.8)
    edges = neo4j_store.get_edges("m1")
    assert any(e.id == edge_id for e in edges)


def test_neo4j_delete_edge(neo4j_store):
    edge_id = neo4j_store.add_edge("m1", "m3", "related_to", 0.5)
    assert neo4j_store.delete_edge(edge_id) is True
    assert neo4j_store.delete_edge(edge_id) is False
