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
"""验证 V2Memory 4 个编排方法 — remember / recall / improve / forget。

零配置默认: SQLite + HashEmbedder + 无 LLM (零 LLM 降级)。
详见 docs/specs/2026-08-04-v2-memory-architecture.md §7 + §8。
"""

from __future__ import annotations

import pytest

from septmuse.configs import default_config
from septmuse.memory import V2Memory
from septmuse.memory.main import Memory


@pytest.fixture
def v2(tmp_path, monkeypatch):
    """V2Memory 实例 (零 LLM, tmp_path 文件 DB)。"""
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "v2_test.db"))
    config = default_config()
    instance = V2Memory(config=config)
    yield instance


class TestV2MemoryCreation:
    """V2Memory 创建验证。"""

    def test_v2_memory_creation(self, v2):
        """V2Memory 创建成功。"""
        assert v2 is not None
        assert v2.mem is not None
        assert v2.llm is None  # 零 LLM 模式

    def test_v2_memory_not_inherit_memory(self):
        """V2Memory 不继承 Memory (组合而非继承)。"""
        assert not issubclass(V2Memory, Memory)

    def test_v2_memory_has_10_components(self, v2):
        """V2Memory 持有 10 个子组件。"""
        assert v2.working_memory is not None
        assert v2.semantic is not None
        assert v2.episodic is not None
        assert v2.procedural is not None
        assert v2.capture is not None
        assert v2.retrieval is not None
        assert v2.token_budget is not None
        assert v2.meta is not None
        assert v2.evolution is not None
        assert v2.causal is not None
        assert v2.forgetting is not None


class TestRemember:
    """remember() 编排方法验证。"""

    def test_remember_captures_text(self, v2):
        """remember 捕获文本并存为情节 raw_log。"""
        result = v2.remember("我喜欢 Python", user_id="alice", session_id="s1")
        assert result["captured"] is True
        assert result["raw_id"] is not None

    def test_remember_zero_llm_no_facts(self, v2):
        """零 LLM 模式: remember 不抽事实 (fact_ids 为空)。"""
        result = v2.remember("我喜欢 Python", user_id="alice", session_id="s1")
        assert result["captured"] is True
        assert result["fact_ids"] == []

    def test_remember_duplicate(self, v2):
        """重复记忆被去重 (不重复存)。"""
        v2.remember("完全相同的文本", user_id="alice", session_id="s1")
        result = v2.remember("完全相同的文本", user_id="alice", session_id="s1")
        assert result["captured"] is False

    def test_remember_with_agent_id(self, v2):
        """带 agent_id 的 remember 更新工作 block。"""
        result = v2.remember("hello world", user_id="alice", agent_id="bot1", session_id="s1")
        assert result["captured"] is True


class TestRecall:
    """recall() 编排方法验证。"""

    def test_recall_returns_memories(self, v2):
        """recall 返回检索结果。"""
        v2.remember("我喜欢 Python 编程", user_id="alice", session_id="s1")
        result = v2.recall("Python", user_id="alice")
        assert "memories" in result
        assert "injected_prompt" in result
        assert "route" in result

    def test_recall_injected_prompt(self, v2):
        """recall 注入 working_memory + procedural prompt。"""
        v2.remember("test content", user_id="alice", session_id="s1")
        result = v2.recall("test", user_id="alice")
        assert "<memory>" in result["injected_prompt"]

    def test_recall_l1_fallback(self, v2):
        """L1 报告首次不存在时跳过 L2 (决策 6)。"""
        v2.remember("some text", user_id="alice", session_id="s1")
        result = v2.recall("some", user_id="alice")
        # 首次使用, improve 还没跑过, strategy 应为 None
        assert result["strategy"] is None

    def test_recall_after_improve_has_strategy(self, v2):
        """improve 跑过后 recall 有 L2 策略 (决策 6)。"""
        v2.remember("some text", user_id="alice", session_id="s1")
        v2.improve(user_id="alice")
        result = v2.recall("some", user_id="alice")
        # improve 跑过后 L1 报告存在, strategy 不应为 None
        assert result["strategy"] is not None


class TestImprove:
    """improve() 编排方法验证。"""

    def test_improve_runs_dream(self, v2):
        """improve 跑 Dream 链接生长。"""
        v2.remember("test memory content", user_id="alice", session_id="s1")
        result = v2.improve(user_id="alice")
        assert "dream" in result
        assert "links_created" in result["dream"]

    def test_improve_persists_coverage(self, v2):
        """improve 持久化 L1 覆盖报告 (SemanticFact tags=["meta","coverage"])。"""
        v2.remember("test memory content", user_id="alice", session_id="s1")
        v2.improve(user_id="alice")
        facts = v2.semantic.get_all_facts(user_id="alice")
        coverage_facts = [
            f for f in facts
            if "meta" in (f.tags or []) and "coverage" in (f.tags or [])
        ]
        assert len(coverage_facts) > 0

    def test_improve_zero_llm_skips_reflect(self, v2):
        """零 LLM 模式: improve 跳过 reflect (rules=0)。"""
        v2.remember("test memory content", user_id="alice", session_id="s1")
        result = v2.improve(user_id="alice")
        assert result["rules"] == 0


class TestForget:
    """forget() 编排方法验证。"""

    def test_forget_invalidates_then_deletes(self, v2):
        """forget 先 invalidate 再 delete (决策 5)。"""
        result = v2.remember("to be forgotten", user_id="alice", session_id="s1")
        memory_id = result["memory_id"]
        forget_result = v2.forget(memory_id, user_id="alice")
        assert forget_result["event"] == "FORGET"
        assert forget_result["invalidated_at"] is not None
        assert forget_result["deleted_at"] is not None
        # 验证记忆已被软删除 (get 返回 None)
        deleted = v2.store.get(memory_id)
        assert deleted is None

    def test_forget_nonexistent_memory(self, v2):
        """forget 不存在的 memory_id 不崩溃。"""
        result = v2.forget("mem-nonexistent", user_id="alice")
        assert result["event"] == "FORGET"


class TestZeroLLMDegradation:
    """零 LLM 降级路径验证 (§8)。"""

    def test_zero_llm_all_methods_no_crash(self, v2):
        """零 LLM 模式下 4 个编排方法都不崩溃。"""
        r1 = v2.remember("zero llm test", user_id="alice", session_id="s1")
        assert r1["captured"] is True

        r2 = v2.recall("zero", user_id="alice")
        assert "memories" in r2

        r3 = v2.improve(user_id="alice")
        assert "dream" in r3

        r4 = v2.forget(r1["memory_id"], user_id="alice")
        assert r4["event"] == "FORGET"

    def test_zero_llm_remember_no_facts(self, v2):
        """零 LLM 模式: remember 只存 raw_log, 不抽事实。"""
        result = v2.remember("no llm extraction", user_id="alice", session_id="s1")
        assert result["fact_ids"] == []
        assert result["raw_id"] is not None
