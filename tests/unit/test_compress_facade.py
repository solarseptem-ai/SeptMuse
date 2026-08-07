"""Memory.compress facade + recall auto_compress 测试."""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def memory(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    return Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))


def test_memory_has_summarizer(memory):
    """Memory.__init__ 创建 self.summarizer."""
    assert hasattr(memory, "summarizer")
    assert memory.summarizer is not None


def test_compress_facade(memory):
    """Memory.compress 委托 Summarizer.compress."""
    # 存 5 条记忆 (用差异化的文本避免 0.95 语义去重)
    facts = [
        "Alice loves Python programming language",
        "Bob works at Google headquarters in London",
        "Cats sleep all day long on the warm sofa",
        "Dogs are very loyal animal companions forever",
        "Earth orbits around the sun every single year",
    ]
    for fact in facts:
        memory.add(fact, user_id="alice", infer=False)
    # buffer_size=3 → 压缩 2 条到摘要
    result = memory.compress(user_id="alice", mode="static", buffer_size=3)
    assert result["compressed"] is True
    assert result["evicted"] == 2
    assert result["kept"] == 3
    assert result["summary_id"] is not None


def test_compress_below_threshold(memory):
    """记忆数 <= buffer_size → 不压缩."""
    memory.add("single fact", user_id="alice", infer=False)
    result = memory.compress(user_id="alice", mode="static", buffer_size=20)
    assert result["compressed"] is False


def test_compress_no_llm_fallback(memory):
    """无 LLM → 拼接摘要降级 (不报错)."""
    for i in range(5):
        memory.add(f"fact {i}", user_id="alice", infer=False)
    result = memory.compress(user_id="alice", buffer_size=2)
    assert result["compressed"] is True
    # summary_id 应该存在 (Summarizer 用拼接降级也存 episode)
    assert result["summary_id"] is not None


def test_recall_auto_compress(memory):
    """recall(auto_compress=True) → 记忆超阈值时自动压缩."""
    # 存 30 条记忆
    for i in range(30):
        memory.add(f"memory fact number {i} about various topics", user_id="alice", infer=False)
    # recall 带 auto_compress=True, 阈值默认 20
    result = memory.recall("topics", user_id="alice", auto_compress=True)
    assert "memories" in result
    # 压缩应触发 (30 > 20)
    # 验证: get_all 记忆数应减少 (旧消息被驱逐)
    all_mems = memory.get_all(user_id="alice")
    mem_list = all_mems.get("results", all_mems) if isinstance(all_mems, dict) else all_mems
    assert len(mem_list) <= 25  # 应该压缩了 (buffer_size 默认 20)


def test_recall_auto_compress_below_threshold(memory):
    """记忆数 <= 阈值 → auto_compress 不触发."""
    memory.add("single fact", user_id="alice", infer=False)
    result = memory.recall("fact", user_id="alice", auto_compress=True)
    assert "memories" in result
