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
"""GraphStore.delete_edge 测试。"""

from __future__ import annotations

import sqlite3
import threading

from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore


def _make_store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "graph.db"))
    lock = threading.Lock()
    return SQLiteGraphStore(conn, lock)


def test_delete_edge_success(tmp_path):
    store = _make_store(tmp_path)
    edge_id = store.add_edge("m1", "m2", "related_to", 0.8)
    assert store.delete_edge(edge_id) is True
    assert store.has_edge("m1", "m2", "related_to") is False


def test_delete_edge_nonexistent_returns_false(tmp_path):
    store = _make_store(tmp_path)
    assert store.delete_edge("nonexistent-edge-id") is False


def test_delete_edge_only_removes_target(tmp_path):
    store = _make_store(tmp_path)
    e1 = store.add_edge("m1", "m2", "related_to", 0.8)
    store.add_edge("m1", "m3", "related_to", 0.5)
    assert store.delete_edge(e1) is True
    assert store.has_edge("m1", "m2", "related_to") is False
    assert store.has_edge("m1", "m3", "related_to") is True
