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
"""记忆更新能力测试 — 长时记忆 update + Block 持久化。"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from septmuse import MemoryConfig
from septmuse.api.rest import create_app
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.models.block import Block, WorkingMemory
from septmuse.storage.sqlite.store import SQLiteMemoryStore
from septmuse.storage.typed_store import TypedMemoryStore


class TestStoreUpdate:
    def test_update_content(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("旧内容", [1.0, 0.0, 0.0], user_id="alice")
        ok = store.update(mid, "新内容", [0.0, 1.0, 0.0])
        assert ok is True
        result = store.get(mid)
        assert result["memory"] == "新内容"
        store.close()

    def test_update_not_found(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        ok = store.update("nonexistent", "x", [1.0])
        assert ok is False
        store.close()

    def test_update_deleted_returns_false(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("内容", [1.0], user_id="alice")
        store.delete(mid)
        ok = store.update(mid, "新", [0.0])
        assert ok is False
        store.close()

    def test_update_history_recorded(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("旧内容", [1.0], user_id="alice")
        store.update(mid, "新内容", [0.0])
        with store._lock:
            cur = store.conn.execute(
                "SELECT event, old_memory, new_memory FROM history WHERE memory_id=? AND event='UPDATE'",
                (mid,),
            )
            rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "UPDATE"
        assert rows[0][1] == "旧内容"
        assert rows[0][2] == "新内容"
        store.close()

    def test_update_metadata_only(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("内容", [1.0], user_id="alice", metadata={"k": "v1"})
        ok = store.update(mid, "内容", [1.0], metadata={"k": "v2"})
        assert ok is True
        result = store.get(mid)
        assert result["metadata"]["k"] == "v2"
        store.close()


def _make_memory(tmp_path):
    return ExperimentalMemory(
        config=MemoryConfig(db_path=str(tmp_path / "test.db")),
        embedder=HashEmbedder(),
    )


class TestFacadeUpdate:
    def test_update_content(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("旧内容", user_id="alice")
        mid = result["results"][0]["id"]
        updated = m.update(mid, "新内容")
        assert updated["event"] == "UPDATE"
        assert updated["memory"] == "新内容"
        # 验证可检索新内容
        hits = m.search("新内容", user_id="alice")
        assert any(h["memory"] == "新内容" for h in hits)

    def test_update_not_found(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update("nonexistent", "x")
        assert result["event"] == "NOT_FOUND"

    def test_update_metadata_only(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("内容", user_id="alice")
        mid = result["results"][0]["id"]
        updated = m.update(mid, metadata={"tag": "important"})
        assert updated["event"] == "UPDATE"
        # 验证 metadata 更新
        item = m.get(mid)
        assert item["metadata"]["tag"] == "important"

    def test_update_re_embedding(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("python programming", user_id="alice")
        mid = result["results"][0]["id"]
        # update 为完全不同的内容
        m.update(mid, "cooking recipes")
        # 旧 query 不应高匹配
        old_hits = m.search("python", user_id="alice", threshold=0.5)
        assert not any(h["id"] == mid for h in old_hits)


class TestBlockStore:
    def test_save_and_get_blocks(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        block = Block(agent_id="agent-1", label="human", value="Name: Alice")
        store.save_block(block)
        loaded = store.get_blocks("agent-1")
        assert len(loaded) == 1
        assert loaded[0].label == "human"
        assert loaded[0].value == "Name: Alice"

    def test_save_block_update_existing(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        block = Block(agent_id="agent-1", label="human", value="v1")
        store.save_block(block)
        block.value = "v2"
        store.save_block(block)
        loaded = store.get_blocks("agent-1")
        assert len(loaded) == 1
        assert loaded[0].value == "v2"

    def test_update_block_value(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.save_block(Block(agent_id="agent-1", label="human", value="old"))
        result = store.update_block_value("agent-1", "human", "new")
        assert result is not None
        assert result.value == "new"

    def test_update_block_value_not_found(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        result = store.update_block_value("agent-1", "nonexistent", "x")
        assert result is None

    def test_delete_block(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.save_block(Block(agent_id="agent-1", label="human", value="x"))
        ok = store.delete_block("agent-1", "human")
        assert ok is True
        assert store.get_blocks("agent-1") == []

    def test_delete_block_not_found(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        ok = store.delete_block("agent-1", "nonexistent")
        assert ok is False

    def test_ensure_default_blocks(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        blocks = store.ensure_default_blocks("agent-1")
        assert len(blocks) == 2
        labels = [b.label for b in blocks]
        assert "human" in labels
        assert "persona" in labels

    def test_ensure_default_blocks_idempotent(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.ensure_default_blocks("agent-1")
        assert len(blocks) == 2


class TestWorkingMemoryPersist:
    def test_store_none_backward_compat(self):
        """store=None 纯内存模式 (向后兼容现有 test_block.py)。"""
        wm = WorkingMemory(agent_id="agent-1")
        wm.update_block_value("human", "value")
        assert wm.get_block("human").value == "value"

    def test_update_block_value_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.update_block_value("human", "persisted value")
        # 重新 load 验证持久化
        reloaded = store.get_blocks("agent-1")
        human = next(b for b in reloaded if b.label == "human")
        assert human.value == "persisted value"

    def test_core_memory_append_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.core_memory_append("human", "Name: Alice")
        reloaded = store.get_blocks("agent-1")
        human = next(b for b in reloaded if b.label == "human")
        assert "Name: Alice" in human.value

    def test_core_memory_replace_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.core_memory_append("human", "Likes: Python, hiking")
        wm.core_memory_replace("human", "hiking", "skiing")
        reloaded = store.get_blocks("agent-1")
        human = next(b for b in reloaded if b.label == "human")
        assert "skiing" in human.value
        assert "hiking" not in human.value

    def test_set_block_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        wm = WorkingMemory("agent-1", store=store)
        wm.set_block(Block(agent_id="agent-1", label="task", value="do x"))
        reloaded = store.get_blocks("agent-1")
        labels = [b.label for b in reloaded]
        assert "task" in labels


class TestFacadeBlock:
    def test_get_blocks_creates_defaults(self, tmp_path):
        m = _make_memory(tmp_path)
        blocks = m.get_blocks("agent-1")
        labels = [b["label"] for b in blocks]
        assert "human" in labels
        assert "persona" in labels

    def test_update_block(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update_block("agent-1", "human", "Name: Alice")
        assert result["event"] == "UPDATE"
        assert result["value"] == "Name: Alice"
        # 验证持久化 — 新 Memory 实例读同一 db
        m2 = _make_memory(tmp_path)
        blocks = m2.get_blocks("agent-1")
        human = next(b for b in blocks if b["label"] == "human")
        assert human["value"] == "Name: Alice"

    def test_core_memory_append(self, tmp_path):
        m = _make_memory(tmp_path)
        m.update_block("agent-1", "human", "Name: Alice")
        result = m.core_memory_append("agent-1", "human", "Likes: Python")
        assert result["event"] == "APPEND"
        assert "Name: Alice" in result["value"]
        assert "Likes: Python" in result["value"]

    def test_core_memory_replace(self, tmp_path):
        m = _make_memory(tmp_path)
        m.update_block("agent-1", "human", "Likes: Python, hiking")
        result = m.core_memory_replace("agent-1", "human", "hiking", "skiing")
        assert result["event"] == "REPLACE"
        assert "skiing" in result["value"]
        assert "hiking" not in result["value"]


def _make_app(tmp_path):
    m = _make_memory(tmp_path)
    return create_app(m), m


class TestRestUpdate:
    def test_put_memory(self, tmp_path):
        app, m = _make_app(tmp_path)
        result = m.add("旧内容", user_id="alice")
        mid = result["results"][0]["id"]
        client = TestClient(app)
        resp = client.put(f"/memories/{mid}", json={"text": "新内容"})
        assert resp.status_code == 200
        assert resp.json()["event"] == "UPDATE"

    def test_put_memory_not_found(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.put("/memories/nonexistent", json={"text": "x"})
        assert resp.status_code == 404

    def test_get_blocks(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.get("/memories/working/blocks/agent-1")
        assert resp.status_code == 200
        labels = [b["label"] for b in resp.json()]
        assert "human" in labels

    def test_put_block(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.put("/memories/working/blocks/agent-1/human", json={"value": "Name: Alice"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "Name: Alice"

    def test_post_append(self, tmp_path):
        app, m = _make_app(tmp_path)
        m.update_block("agent-1", "human", "Name: Alice")
        client = TestClient(app)
        resp = client.post("/memories/working/blocks/agent-1/human/append", json={"content": "Likes: Python"})
        assert resp.status_code == 200
        assert "Likes: Python" in resp.json()["value"]

    def test_post_replace(self, tmp_path):
        app, m = _make_app(tmp_path)
        m.update_block("agent-1", "human", "Likes: Python, hiking")
        client = TestClient(app)
        resp = client.post(
            "/memories/working/blocks/agent-1/human/replace",
            json={"old_content": "hiking", "new_content": "skiing"},
        )
        assert resp.status_code == 200
        assert "skiing" in resp.json()["value"]


class TestHistory:
    def test_get_history_after_add(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("content", user_id="alice")
        mid = result["results"][0]["id"]
        history = m.get_history(mid)
        assert len(history) >= 1
        assert history[0]["event"] == "ADD"

    def test_get_history_after_update(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("old", user_id="alice")
        mid = result["results"][0]["id"]
        m.update(mid, "new")
        history = m.get_history(mid)
        events = [h["event"] for h in history]
        assert "ADD" in events
        assert "UPDATE" in events

    def test_get_history_empty(self, tmp_path):
        m = _make_memory(tmp_path)
        history = m.get_history("nonexistent")
        assert history == []

    def test_rest_get_history(self, tmp_path):
        m = _make_memory(tmp_path)
        app = create_app(m)
        result = m.add("content", user_id="alice")
        mid = result["results"][0]["id"]
        client = TestClient(app)
        resp = client.get(f"/memories/{mid}/history")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_cli_history(self, tmp_path, monkeypatch, capsys):
        from septmuse.cli.main import main

        db = str(tmp_path / "t.db")
        monkeypatch.setattr(sys, "argv", ["septmuse", "init", "--user", "alice", "--db-path", db])
        main()
        monkeypatch.setattr(sys, "argv", ["septmuse", "add", "content", "--user", "alice", "--db-path", db])
        main()
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["septmuse", "add", "content2", "--user", "alice", "--db-path", db])
        main()
        out2 = capsys.readouterr().out
        mid = json.loads(out2).get("memory_id")
        monkeypatch.setattr(sys, "argv", ["septmuse", "history", mid, "--db-path", db])
        main()
        out3 = capsys.readouterr().out
        history = json.loads(out3)
        assert len(history) >= 1


class TestTypedUpdate:
    def test_update_fact(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_fact("alice", "likes", "python", user_id="alice")
        fid = result["id"]
        updated = m.update_fact(fid, subject="alice", predicate="likes", object="rust", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_episode_content(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_episode("old event", user_id="alice")
        eid = result["id"]
        updated = m.update_episode(eid, content="new event", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_rule(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_rule("old rule", user_id="alice")
        rid = result["id"]
        updated = m.update_rule(rid, rule="new rule", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_fact_not_found(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update_fact("nonexistent", subject="x", predicate="y", object="z", user_id="alice")
        assert result["event"] == "NOT_FOUND"
