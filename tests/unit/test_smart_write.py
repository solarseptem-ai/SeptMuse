"""Tests for Smart Write integration in remember()."""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest


@pytest.fixture
def tmp_memory():
    """Create Memory with temp DB."""
    db = tempfile.mktemp(suffix=".db")
    os.environ["SEPTMUSE_DB_PATH"] = db
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs.base import MemoryConfig
    from septmuse.embedders.hash import HashEmbedder
    from septmuse.memory.main import Memory

    mem = Memory(config=MemoryConfig(db_path=db), embedder=HashEmbedder())
    yield mem
    for f in [db, db + "-wal", db + "-shm"]:
        with contextlib.suppress(OSError):
            os.unlink(f)


def test_remember_returns_importance(tmp_memory):
    """remember 返回 importance 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "importance" in result
    assert isinstance(result["importance"], (int, float))
    assert result["importance"] >= 0.0


def test_remember_returns_types(tmp_memory):
    """remember 返回 types 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "types" in result
    assert isinstance(result["types"], list)
    assert len(result["types"]) > 0


def test_remember_returns_duplicate(tmp_memory):
    """remember 返回 duplicate 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "duplicate" in result
    assert result["duplicate"] is None  # 首次写入无重复


def test_remember_returns_conflicts(tmp_memory):
    """remember 返回 conflicts 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "conflicts" in result
    assert isinstance(result["conflicts"], list)


def test_remember_low_importance_skipped(tmp_memory):
    """低价值文本跳过记忆存储，但 raw_log 仍存。"""
    result = tmp_memory.remember("嗯好的", user_id="alice")
    assert result["captured"] is False or result.get("importance", 1.0) < 0.5
    assert result["raw_id"] is not None  # raw_log 仍存


def test_remember_semantic_dedup(tmp_memory):
    """重复内容第二次写入返回 duplicate。"""
    tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    # SHA256 或语义去重应捕获
    assert result["captured"] is False or result.get("duplicate") is not None or result["memory_id"] is None
