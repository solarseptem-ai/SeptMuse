"""遗忘曲线参数化测试."""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def typed_store(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

    return TypedMemoryStore(db_path=str(tmp_path / "test.db"))


def test_decay_default_half_life(typed_store):
    """默认 half_life_days=None → 用原公式 (向后兼容)."""
    from septmuse.models.strength import MemoryStrength

    s = MemoryStrength(memory_id="m1", user_id="alice", strength=1.0, base_value=0.5)
    now = datetime.now(timezone.utc)
    decayed = s.decay(now)
    assert 0.0 <= decayed <= 1.0


def test_decay_fast_half_life(typed_store):
    """half_life_days=1.0 → 1 天后衰减到 ~0.5."""
    from septmuse.models.strength import MemoryStrength

    now = datetime.now(timezone.utc)
    past = now - timedelta(days=1)  # 1 天前
    s = MemoryStrength(
        memory_id="m2", user_id="alice", strength=1.0, base_value=0.5, last_accessed=past
    )
    decayed = s.decay(now, half_life_days=1.0)
    # 1 天 = 1 个半衰期 → 衰减到 0.5 左右
    assert 0.4 <= decayed <= 0.6


def test_decay_permanent(typed_store):
    """half_life_days=inf → 不衰减 (permanent)."""
    from septmuse.models.strength import MemoryStrength

    now = datetime.now(timezone.utc)
    past = now - timedelta(days=365)  # 1 年前
    s = MemoryStrength(
        memory_id="m3", user_id="alice", strength=1.0, base_value=0.5, last_accessed=past
    )
    decayed = s.decay(now, half_life_days=math.inf)
    assert decayed == 1.0  # 永不衰减


def test_decay_slow_half_life(typed_store):
    """half_life_days=30.0 → 1 天后几乎不衰减."""
    from septmuse.models.strength import MemoryStrength

    now = datetime.now(timezone.utc)
    past = now - timedelta(days=1)  # 1 天前
    s = MemoryStrength(
        memory_id="m4", user_id="alice", strength=1.0, base_value=0.5, last_accessed=past
    )
    decayed = s.decay(now, half_life_days=30.0)
    # 30 天半衰期, 1 天后应该衰减很少 (>0.95)
    assert decayed > 0.95


def test_forgetting_retriever_half_life(typed_store):
    """ForgettingRetriever 用配置的 half_life_days."""
    from septmuse.retrieval.forgetting import ForgettingRetriever

    now = datetime.now(timezone.utc)
    typed_store.get_or_create_strength("m5", user_id="alice")
    retriever = ForgettingRetriever(typed_store, half_life_days=1.0)
    results = retriever.apply_strength(
        [{"id": "m5", "memory": "test", "score": 1.0}], user_id="alice", now=now
    )
    assert len(results) > 0


def test_memory_config_half_life():
    """MemoryConfig 有 forgetting_half_life_days 字段."""
    from septmuse.configs import MemoryConfig

    cfg = MemoryConfig()
    assert hasattr(cfg, "forgetting_half_life_days")
    assert cfg.forgetting_half_life_days == 7.0  # 默认 7 天


def test_memory_uses_config_half_life(tmp_path):
    """Memory 创建 ForgettingRetriever 时用 config 的 half_life_days."""
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory

    cfg = MemoryConfig(db_path=str(tmp_path / "test.db"))
    mem = Memory(config=cfg)
    assert mem.forgetting is not None
    assert hasattr(mem.forgetting, "half_life_days")
    assert mem.forgetting.half_life_days == 7.0
