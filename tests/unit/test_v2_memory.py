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
"""验证 Memory 编排方法 — remember / recall / improve / forget。

零配置默认: SQLite + HashEmbedder + 无 LLM (零 LLM 降级)。
"""

from __future__ import annotations

import pytest

from septmuse.configs import default_config
from septmuse.memory.main import Memory


@pytest.fixture
def mem(tmp_path, monkeypatch):
    """Memory 实例 (零 LLM, tmp_path 文件 DB)。"""
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "v2_test.db"))
    config = default_config()
    instance = Memory(config=config)
    yield instance


class TestRemember:
    """remember() 编排方法验证。"""

    def test_remember_captures_text(self, mem):
        """remember 捕获文本并存为情节 raw_log。"""
        result = mem.remember("我喜欢 Python", user_id="alice", session_id="s1")
        assert result["captured"] is True
        assert result["raw_id"] is not None

    def test_remember_zero_llm_no_facts(self, mem):
        """零 LLM 模式: remember 不抽事实 (fact_ids 为空)。"""
        result = mem.remember("我喜欢 Python", user_id="alice", session_id="s1")
        assert result["captured"] is True
        assert result["fact_ids"] == []

    def test_remember_duplicate(self, mem):
        """重复记忆被去重 (不重复存)。"""
        mem.remember("完全相同的文本", user_id="alice", session_id="s1")
        result = mem.remember("完全相同的文本", user_id="alice", session_id="s1")
        assert result["captured"] is False

    def test_remember_with_agent_id(self, mem):
        """带 agent_id 的 remember 更新工作 block。"""
        result = mem.remember("hello world 42", user_id="alice", agent_id="bot1", session_id="s1")
        assert result["captured"] is True


class TestRecall:
    """recall() 编排方法验证。"""

    def test_recall_returns_memories(self, mem):
        """recall 返回检索结果。"""
        mem.remember("我喜欢 Python 编程", user_id="alice", session_id="s1")
        result = mem.recall("Python", user_id="alice")
        assert "memories" in result
        assert "injected_prompt" in result
        assert "route" in result

    def test_recall_injected_prompt(self, mem):
        """recall 注入 working_memory + procedural prompt。"""
        mem.remember("test content", user_id="alice", session_id="s1")
        result = mem.recall("test", user_id="alice")
        assert "<memory>" in result["injected_prompt"]

    def test_recall_l1_fallback(self, mem):
        """L1 报告首次不存在时跳过 L2 (决策 6)。"""
        mem.remember("some text", user_id="alice", session_id="s1")
        result = mem.recall("some", user_id="alice")
        assert result["strategy"] is None

    def test_recall_after_improve_has_strategy(self, mem):
        """improve 跑过后 recall 有 L2 策略 (决策 6)。"""
        mem.remember("some text", user_id="alice", session_id="s1")
        mem.improve(user_id="alice")
        result = mem.recall("some", user_id="alice")
        assert result["strategy"] is not None


class TestImprove:
    """improve() 编排方法验证。"""

    def test_improve_runs_dream(self, mem):
        """improve 跑 Dream 链接生长。"""
        mem.remember("test memory content", user_id="alice", session_id="s1")
        result = mem.improve(user_id="alice")
        assert "dream" in result
        assert "links_created" in result["dream"]

    def test_improve_persists_coverage(self, mem):
        """improve 持久化 L1 覆盖报告 (SemanticFact tags=["meta","coverage"])。"""
        mem.remember("test memory content", user_id="alice", session_id="s1")
        mem.improve(user_id="alice")
        facts = mem.semantic.get_all_facts(user_id="alice")
        coverage_facts = [f for f in facts if "meta" in (f.tags or []) and "coverage" in (f.tags or [])]
        assert len(coverage_facts) > 0

    def test_improve_zero_llm_skips_reflect(self, mem):
        """零 LLM 模式: improve 跳过 reflect (rules=0)。"""
        mem.remember("test memory content", user_id="alice", session_id="s1")
        result = mem.improve(user_id="alice")
        assert result["rules"] == 0


class TestForget:
    """forget() 编排方法验证。"""

    def test_forget_invalidates_then_deletes(self, mem):
        """forget 先 invalidate 再 delete (决策 5)。"""
        result = mem.remember("to be forgotten 42", user_id="alice", session_id="s1")
        memory_id = result["memory_id"]
        forget_result = mem.forget(memory_id, user_id="alice")
        assert forget_result["event"] == "FORGET"
        assert forget_result["invalidated_at"] is not None
        assert forget_result["deleted_at"] is not None
        deleted = mem.store.get(memory_id)
        assert deleted is None

    def test_forget_nonexistent_memory(self, mem):
        """forget 不存在的 memory_id 不崩溃。"""
        result = mem.forget("mem-nonexistent", user_id="alice")
        assert result["event"] == "FORGET"


class TestZeroLLMDegradation:
    """零 LLM 降级路径验证 (§8)。"""

    def test_zero_llm_all_methods_no_crash(self, mem):
        """零 LLM 模式下 4 个编排方法都不崩溃。"""
        r1 = mem.remember("zero llm test 42", user_id="alice", session_id="s1")
        assert r1["captured"] is True

        r2 = mem.recall("zero", user_id="alice")
        assert "memories" in r2

        r3 = mem.improve(user_id="alice")
        assert "dream" in r3

        r4 = mem.forget(r1["memory_id"], user_id="alice")
        assert r4["event"] == "FORGET"

    def test_zero_llm_remember_no_facts(self, mem):
        """零 LLM 模式: remember 只存 raw_log, 不抽事实。"""
        result = mem.remember("no llm extraction", user_id="alice", session_id="s1")
        assert result["fact_ids"] == []
        assert result["raw_id"] is not None
