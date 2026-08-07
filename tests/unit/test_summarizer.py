"""P2-Task 3: 消息压缩 Summarizer 单元测试。

验收标准:
- 50 条消息压缩到 20 条 + 1 条摘要
- 摘要保留关键信息 (测试用 LLM mock 验证)
- ≥10 个单元测试
"""

from __future__ import annotations

from pathlib import Path

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.evolution.summarizer import Summarizer
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM


class StubLLM(LLM):
    """测试用 LLM stub。"""

    def __init__(self, response: str = ""):
        self._response = response or "Summary of conversation"

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_summarizer.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


@pytest.fixture()
def memory_with_llm(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=tmp_db),
        embedder=HashEmbedder(dim=128),
        llm=StubLLM("LLM summary of messages"),
    )


@pytest.fixture()
def memory_with_50_msgs(memory: ExperimentalMemory) -> ExperimentalMemory:
    """预填 50 条消息。"""
    for i in range(50):
        memory.add(f"message number {i}", user_id="u1")
    return memory


class TestStaticBuffer:
    def test_compress_50_to_20(self, memory_with_50_msgs: ExperimentalMemory):
        """验收: 50 条消息压缩到 20 条 + 1 条摘要。"""
        result = memory_with_50_msgs.compress(user_id="u1", mode="static", buffer_size=20)
        assert result["compressed"] is True
        assert result["evicted"] == 30
        assert result["kept"] == 20
        assert result["summary_id"] is not None

    def test_compress_creates_summary_episode(self, memory_with_50_msgs: ExperimentalMemory):
        """压缩后创建 EpisodicEvent (event_type=summary)。"""
        result = memory_with_50_msgs.compress(user_id="u1", mode="static", buffer_size=20)
        episode = memory_with_50_msgs.typed_store.get_episodes(user_id="u1", limit=100)
        summaries = [e for e in episode if e.event_type == "summary"]
        assert len(summaries) == 1
        assert summaries[0].id == result["summary_id"]

    def test_compress_deletes_evicted(self, memory_with_50_msgs: ExperimentalMemory):
        """压缩后 evicted 消息被软删除。"""
        before = memory_with_50_msgs.get_all(user_id="u1")
        assert len(before["results"]) == 50
        memory_with_50_msgs.compress(user_id="u1", mode="static", buffer_size=20)
        after = memory_with_50_msgs.get_all(user_id="u1")
        assert len(after["results"]) == 20

    def test_compress_under_buffer_skipped(self, memory: ExperimentalMemory):
        """消息数 <= buffer_size 时不压缩。"""
        memory.add("msg 1", user_id="u1")
        memory.add("msg 2", user_id="u1")
        result = memory.compress(user_id="u1", mode="static", buffer_size=20)
        assert result["compressed"] is False
        assert result["evicted"] == 0

    def test_compress_empty_returns_not_compressed(self, memory: ExperimentalMemory):
        """空记忆库不压缩。"""
        result = memory.compress(user_id="u1", mode="static", buffer_size=20)
        assert result["compressed"] is False


class TestPartialEvict:
    def test_partial_evict_removes_30_percent(self, memory_with_50_msgs: ExperimentalMemory):
        """partial 模式驱逐 30%。"""
        result = memory_with_50_msgs.compress(user_id="u1", mode="partial", buffer_size=20)
        assert result["compressed"] is True
        assert result["evicted"] == 15
        assert result["kept"] == 35

    def test_partial_evict_minimum_1(self, memory: ExperimentalMemory):
        """partial 模式最少驱逐 1 条。"""
        for i in range(3):
            memory.add(f"msg {i}", user_id="u1")
        result = memory.compress(user_id="u1", mode="partial", buffer_size=1)
        assert result["evicted"] >= 1


class TestLLMSummary:
    def test_llm_summary_used(self, memory_with_llm: ExperimentalMemory):
        """验收: LLM 摘要保留关键信息 (用 mock 验证)。"""
        for i in range(25):
            memory_with_llm.add(f"important fact {i}", user_id="u1")
        result = memory_with_llm.compress(user_id="u1", mode="static", buffer_size=10)
        assert result["compressed"] is True

        episodes = memory_with_llm.typed_store.get_episodes(user_id="u1", limit=100)
        summaries = [e for e in episodes if e.event_type == "summary"]
        assert len(summaries) == 1
        assert summaries[0].content == "LLM summary of messages"

    def test_no_llm_uses_concatenation(self, memory_with_50_msgs: ExperimentalMemory):
        """无 LLM 时用拼接降级。"""
        memory_with_50_msgs.compress(user_id="u1", mode="static", buffer_size=20)
        episodes = memory_with_50_msgs.typed_store.get_episodes(user_id="u1", limit=100)
        summaries = [e for e in episodes if e.event_type == "summary"]
        assert len(summaries) == 1
        assert "[Summary]" in summaries[0].content

    def test_llm_failure_falls_back(self, memory: ExperimentalMemory):
        """LLM 调用失败时回退拼接。"""

        class FailLLM(LLM):
            def _complete(self, system_prompt: str, user_prompt: str) -> str:
                raise RuntimeError("LLM down")

        summarizer = Summarizer(memory.store, memory.typed_store, llm=FailLLM())
        for i in range(25):
            memory.add(f"msg {i}", user_id="u1")
        result = summarizer.compress(user_id="u1", mode="static", buffer_size=10)
        assert result["compressed"] is True
        episodes = memory.typed_store.get_episodes(user_id="u1", limit=100)
        summaries = [e for e in episodes if e.event_type == "summary"]
        assert len(summaries) == 1
        assert "[Summary]" in summaries[0].content


class TestUserIsolation:
    def test_compress_only_affects_user(self, memory: ExperimentalMemory):
        """压缩只影响指定用户。"""
        for i in range(25):
            memory.add(f"u1 msg {i}", user_id="u1")
        for i in range(25):
            memory.add(f"u2 msg {i}", user_id="u2")

        memory.compress(user_id="u1", mode="static", buffer_size=10)

        u1_count = len(memory.get_all(user_id="u1")["results"])
        u2_count = len(memory.get_all(user_id="u2")["results"])
        assert u1_count == 10
        assert u2_count == 25


class TestSummarizerDirect:
    def test_summarizer_directly(self, memory: ExperimentalMemory):
        """直接用 Summarizer 测试。"""
        for i in range(25):
            memory.add(f"msg {i}", user_id="u1")
        summarizer = Summarizer(memory.store, memory.typed_store, llm=None)
        result = summarizer.compress(user_id="u1", mode="static", buffer_size=10)
        assert result["compressed"] is True
        assert result["evicted"] == 15
