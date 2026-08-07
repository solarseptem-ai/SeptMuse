"""recall 画像注入测试."""
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


def test_recall_inject_profile(memory):
    """recall(inject_profile=True) → injected_prompt 含画像."""
    memory.typed_store.add_fact("user", "name", "Alice", user_id="alice", confidence=0.9)
    memory.typed_store.add_fact("user", "likes", "Python", user_id="alice")
    memory.remember("I like Python", user_id="alice")
    result = memory.recall("what do I like", user_id="alice", inject_profile=True)
    assert "Alice" in result["injected_prompt"]
    assert "Python" in result["injected_prompt"]


def test_recall_no_profile_by_default(memory):
    """默认 inject_profile=False, 不注入画像."""
    memory.typed_store.add_fact("user", "name", "Alice", user_id="alice")
    memory.remember("test", user_id="alice")
    result = memory.recall("test", user_id="alice")
    assert "Alice" not in (result.get("injected_prompt") or "")


def test_recall_profile_empty_user(memory):
    """无记忆用户 + inject_profile=True → injected_prompt 不含画像段 (不报错)."""
    result = memory.recall("anything", user_id="nobody", inject_profile=True)
    # 不报错, injected_prompt 可能为空
    assert "injected_prompt" in result


def test_profile_to_prompt_format(memory):
    """_profile_to_prompt 输出格式正确."""
    memory.typed_store.add_fact("user", "name", "Alice", user_id="alice")
    memory.typed_store.add_fact("user", "occupation", "Engineer", user_id="alice")
    memory.typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    prompt = memory._profile_to_prompt(profile)
    assert "Alice" in prompt
    assert "Engineer" in prompt
    assert "Python" in prompt
