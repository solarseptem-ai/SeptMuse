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
"""双时态建模单元测试 (借鉴 graphiti EntityEdge bitemporal fields)。"""

from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.experimental import ExperimentalMemory
from septmuse.storage.sqlite.store import SQLiteMemoryStore


class TestTemporalMigration:
    def test_new_db_has_temporal_columns(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(memories)")}
        assert "valid_at" in cols
        assert "invalid_at" in cols
        assert "expired_at" in cols
        store.close()

    def test_idempotent_migration(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store._migrate_add_temporal_columns()
        store._migrate_add_temporal_columns()
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(memories)")}
        assert "valid_at" in cols
        store.close()


class TestAddWithValidAt:
    def test_add_with_valid_at(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("Alice works at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        row = store.conn.execute("SELECT valid_at, invalid_at, expired_at FROM memories WHERE id=?", (mid,)).fetchone()
        assert row[0] == "2024-01-01"
        assert row[1] is None
        assert row[2] is None
        store.close()

    def test_add_without_valid_at(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("hello world", [1.0, 0.0], user_id="u1")
        row = store.conn.execute("SELECT valid_at FROM memories WHERE id=?", (mid,)).fetchone()
        assert row[0] is None
        store.close()


class TestGetTemporalValid:
    def test_returns_memories_valid_at_reference_time(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("Alice at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.add("Alice at Apple", [1.0, 0.0], user_id="u1", valid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 1
        assert "Google" in results[0]["memory"]

    def test_returns_null_valid_at(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("no time constraint", [1.0, 0.0], user_id="u1")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 1

    def test_excludes_invalidated(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("old fact", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.invalidate(mid, invalid_at="2024-06-01")
        results = store.get_temporal_valid("2024-07-01", user_id="u1")
        assert len(results) == 0

    def test_includes_still_valid(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("current fact", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.invalidate(mid, invalid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 1

    def test_excludes_future_valid_at(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("future fact", [1.0, 0.0], user_id="u1", valid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 0


class TestInvalidate:
    def test_invalidate_sets_columns(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("Alice at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        result = store.invalidate(mid, invalid_at="2025-01-01")
        assert result["event"] == "INVALIDATE"
        assert result["invalid_at"] == "2025-01-01"
        row = store.conn.execute("SELECT invalid_at, expired_at FROM memories WHERE id=?", (mid,)).fetchone()
        assert row[0] == "2025-01-01"
        assert row[1] is not None

    def test_invalidate_default_time(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("test", [1.0, 0.0], user_id="u1")
        result = store.invalidate(mid)
        assert result["invalid_at"] is not None
        assert result["expired_at"] is not None

    def test_invalidate_not_found(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        result = store.invalidate("nonexistent-id")
        assert result["event"] == "NOT_FOUND"

    def test_invalidate_does_not_delete(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("test", [1.0, 0.0], user_id="u1")
        store.invalidate(mid)
        m = store.get(mid)
        assert m is not None
        assert m["id"] == mid


class TestMemoryAddValidAt:
    def test_add_with_valid_at(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
        assert len(result["results"]) == 1
        mid = result["results"][0]["id"]
        mem = m.get(mid)
        assert mem is not None

    def test_add_without_valid_at(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("hello world", user_id="u1")
        assert len(result["results"]) == 1


class TestMemoryInvalidate:
    def test_invalidate_existing(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice at Google", user_id="u1", valid_at="2024-01-01")
        mid = result["results"][0]["id"]
        inv = m.invalidate(mid, invalid_at="2025-01-01")
        assert inv["event"] == "INVALIDATE"
        assert inv["invalid_at"] == "2025-01-01"

    def test_invalidate_not_found(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        inv = m.invalidate("nonexistent")
        assert inv["event"] == "NOT_FOUND"

    def test_invalidate_default_time(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("test", user_id="u1")
        mid = result["results"][0]["id"]
        inv = m.invalidate(mid)
        assert inv["invalid_at"] is not None
        assert inv["expired_at"] is not None


class TestMemorySearchAt:
    def test_search_at_returns_valid_facts(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
        m.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")
        results = m.search_at("2024-06-01", "Alice", user_id="u1")
        assert len(results) >= 1
        assert any("Google" in r["memory"] for r in results)

    def test_search_at_excludes_future(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("future fact", user_id="u1", valid_at="2025-01-01")
        results = m.search_at("2024-06-01", "future", user_id="u1")
        assert len(results) == 0

    def test_search_at_includes_null_valid_at(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("no time constraint", user_id="u1")
        results = m.search_at("2024-06-01", "time", user_id="u1")
        assert len(results) >= 1

    def test_search_at_excludes_invalidated(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("old fact", user_id="u1", valid_at="2024-01-01")
        mid = result["results"][0]["id"]
        m.invalidate(mid, invalid_at="2024-06-01")
        results = m.search_at("2024-07-01", "old", user_id="u1")
        assert len(results) == 0


class TestCLIValidAt:
    def test_cli_add_with_valid_at(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["add", "hello", "--user-id", "u1", "--valid-at", "2024-01-01"])
        assert args.valid_at == "2024-01-01"

    def test_cli_invalidate_command(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["invalidate", "mem-123"])
        assert args.memory_id == "mem-123"


class TestRESTInvalidate:
    def test_rest_invalidate(self, tmp_path):
        from fastapi.testclient import TestClient

        from septmuse.api.rest import create_app
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "rest.db"))
        app = create_app(config)
        client = TestClient(app)

        resp = client.post("/memories", json={"messages": "hello", "user_id": "u1"})
        mid = resp.json()["results"][0]["id"]

        resp = client.post(f"/memories/{mid}/invalidate", json={"invalid_at": "2025-01-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["event"] == "INVALIDATE"
        assert data["invalid_at"] == "2025-01-01"
