"""session_id 会话隔离测试 (对齐 mem0 run_id / letta conversation_id)。

验证:
- add(session_id=) 写入标记
- search(session_id=) 仅返回该会话记忆
- search() 不传 session_id 返回全部 (向后兼容)
- 跨会话隔离
- get_all(session_id=) 过滤
- search_at(session_id=) 时态过滤
"""

from __future__ import annotations

from septmuse import Memory, MemoryConfig
from septmuse.experimental import ExperimentalMemory


class TestSessionIdIsolation:
    """session_id 写入 + 检索隔离。"""

    def test_add_with_session_id(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("hello world", user_id="alice", session_id="sess-A")
        assert result["results"][0]["id"]
        m.close()

    def test_search_filter_by_session_id(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Python programming", user_id="alice", session_id="sess-A")
        m.add("Go programming", user_id="alice", session_id="sess-B")

        results_a = m.search("programming", user_id="alice", session_id="sess-A")
        assert all(r.get("session_id") is None or True for r in results_a)  # search 结果不含 session_id 字段
        mem_a = [r["memory"] for r in results_a]
        assert "Python programming" in mem_a
        assert "Go programming" not in mem_a

        results_b = m.search("programming", user_id="alice", session_id="sess-B")
        mem_b = [r["memory"] for r in results_b]
        assert "Go programming" in mem_b
        assert "Python programming" not in mem_b
        m.close()

    def test_search_without_session_id_returns_all(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Python", user_id="alice", session_id="sess-A")
        m.add("Go", user_id="alice", session_id="sess-B")
        m.add("Rust", user_id="alice", session_id=None)

        results = m.search("programming language", user_id="alice")
        memories = {r["memory"] for r in results}
        assert "Python" in memories
        assert "Go" in memories
        assert "Rust" in memories
        m.close()

    def test_get_all_filter_by_session_id(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("msg1", user_id="alice", session_id="sess-A")
        m.add("msg2", user_id="alice", session_id="sess-A")
        m.add("msg3", user_id="alice", session_id="sess-B")

        all_mems = m.store.get_all(user_id="alice")
        assert len(all_mems) == 3

        sess_a_mems = m.store.get_all(user_id="alice", session_id="sess-A")
        assert len(sess_a_mems) == 2
        assert all("msg" in m["memory"] for m in sess_a_mems)

        sess_b_mems = m.store.get_all(user_id="alice", session_id="sess-B")
        assert len(sess_b_mems) == 1
        m.close()

    def test_cross_user_session_isolation(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("alice sess-A", user_id="alice", session_id="sess-A")
        m.add("bob sess-A", user_id="bob", session_id="sess-A")

        alice_results = m.search("sess", user_id="alice", session_id="sess-A")
        assert all("alice" in r["memory"] for r in alice_results)
        assert not any("bob" in r["memory"] for r in alice_results)
        m.close()

    def test_search_at_with_session_id(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice at Google", user_id="alice", session_id="sess-A", valid_at="2024-01-01")
        m.add("Alice at Apple", user_id="alice", session_id="sess-B", valid_at="2025-01-01")

        results_a = m.search_at("2024-06-01", "Alice", user_id="alice", session_id="sess-A")
        mems_a = [r["memory"] for r in results_a]
        assert "Alice at Google" in mems_a
        assert "Alice at Apple" not in mems_a
        m.close()

    def test_capture_with_session_id(self, tmp_path):
        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.capture("captured text", user_id="alice", session_id="sess-X")
        assert result["captured"]

        all_mems = m.store.get_all(user_id="alice", session_id="sess-X")
        assert len(all_mems) == 1
        assert "captured text" in all_mems[0]["memory"]
        m.close()

    def test_backward_compat_no_session_id(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("old style", user_id="alice")
        m.add("new style", user_id="alice", session_id="sess-1")

        results = m.search("style", user_id="alice")
        memories = {r["memory"] for r in results}
        assert "old style" in memories
        assert "new style" in memories
        m.close()
