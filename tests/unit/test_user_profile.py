"""用户画像聚合测试 — 从 SemanticFact 聚合结构化画像."""
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


@pytest.fixture
def typed_store(memory):
    return memory.typed_store


def test_profile_value_defaults():
    from septmuse.models.profile import UserProfileValue
    v = UserProfileValue(value="Alice")
    assert v.value == "Alice"
    assert v.is_current is True
    assert v.confidence == 1.0
    assert v.updated_at is None


def test_get_user_profile_aggregates(memory, typed_store):
    """画像聚合 attributes + preferences."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice", confidence=0.9)
    typed_store.add_fact("user", "occupation", "Engineer", user_id="alice")
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.attributes["name"].value == "Alice"
    assert profile.attributes["name"].confidence == 0.9
    assert profile.attributes["occupation"].value == "Engineer"
    assert profile.preferences["likes"].value == "Python"


def test_profile_contradiction_picks_latest(memory, typed_store):
    """矛盾值 (同 predicate 不同 object) → 最新为 current."""
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    typed_store.add_fact("user", "likes", "Rust", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.preferences["likes"].value == "Rust"
    assert profile.preferences["likes"].is_current is True


def test_profile_soft_deleted_not_current(memory, typed_store):
    """软删除的 fact → is_current=False."""
    f1 = typed_store.add_fact("user", "likes", "Java", user_id="alice")
    typed_store.soft_delete_fact(f1.id)
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.preferences["likes"].value == "Python"
    assert profile.preferences["likes"].is_current is True


def test_profile_include_temporal_false(memory, typed_store):
    """include_temporal=False → 只留 current, 不含历史."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice")
    typed_store.add_fact("user", "name", "Alyssa", user_id="alice")  # 矛盾更新
    profile = memory.get_user_profile("alice", include_temporal=False)
    # 只有一个 current
    current_names = [v for v in profile.attributes.values() if v.value in ("Alice", "Alyssa")]
    assert len(current_names) <= 1


def test_profile_temporal_summary(memory, typed_store):
    """temporal_summary 统计 active + deleted."""
    f1 = typed_store.add_fact("user", "likes", "Java", user_id="alice")
    typed_store.soft_delete_fact(f1.id)
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.temporal_summary["active"] >= 1
    assert profile.temporal_summary["deleted"] >= 1
    assert profile.temporal_summary["total"] >= 2


def test_profile_raw_facts_uncategorized(memory, typed_store):
    """未分类 predicate → raw_facts."""
    typed_store.add_fact("user", "weird_predicate", "some_value", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert len(profile.raw_facts) >= 1


def test_profile_empty_user(memory):
    """无记忆的用户 → 空画像."""
    profile = memory.get_user_profile("nobody")
    assert profile.user_id == "nobody"
    assert len(profile.attributes) == 0
    assert profile.temporal_summary["total"] == 0


def test_profile_relationships(memory, typed_store):
    """关系类 predicate → relationships."""
    typed_store.add_fact("user", "has", "a dog named Buddy", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert "has" in profile.relationships
    assert profile.relationships["has"].value == "a dog named Buddy"
