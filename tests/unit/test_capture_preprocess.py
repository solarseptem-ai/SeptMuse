"""CapturePipeline.preprocess 测试 — 去重+脱敏不写 store."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def capture_pipeline(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.capture.pipeline import CapturePipeline
    from septmuse.embedders.hash import HashEmbedder

    mock_store = MagicMock()
    embedder = HashEmbedder()
    return CapturePipeline(mock_store, embedder), mock_store


def test_preprocess_allows_new_text(capture_pipeline):
    cp, _ = capture_pipeline
    r = cp.preprocess("hello world", user_id="alice")
    assert r.allowed is True
    assert r.stored_text == "hello world"


def test_preprocess_dedup_blocks_duplicate(capture_pipeline):
    """同一用户同文本第二次被去重拒绝."""
    cp, _ = capture_pipeline
    r1 = cp.preprocess("hello world", user_id="alice")
    assert r1.allowed is True
    r2 = cp.preprocess("hello world", user_id="alice")
    assert r2.allowed is False


def test_preprocess_no_write_to_store(capture_pipeline):
    """preprocess 不调 store.add (避免与 Memory.add 双写)."""
    cp, mock_store = capture_pipeline
    cp.preprocess("hello world", user_id="alice")
    assert mock_store.add.call_count == 0


def test_preprocess_empty_text(capture_pipeline):
    cp, _ = capture_pipeline
    r = cp.preprocess("", user_id="alice")
    assert r.allowed is False
    assert "empty" in (r.reason or "").lower()


def test_preprocess_whitespace_text(capture_pipeline):
    cp, _ = capture_pipeline
    r = cp.preprocess("   ", user_id="alice")
    assert r.allowed is False


def test_preprocess_returns_text_hash(capture_pipeline):
    """preprocess 返回 text_hash (去重 hash)."""
    cp, _ = capture_pipeline
    r = cp.preprocess("hello world", user_id="alice")
    assert r.text_hash is not None
    assert len(r.text_hash) > 0


def test_preprocess_different_users_not_deduped(capture_pipeline):
    """不同用户同文本不去重."""
    cp, _ = capture_pipeline
    r1 = cp.preprocess("hello", user_id="alice")
    r2 = cp.preprocess("hello", user_id="bob")
    assert r1.allowed is True
    assert r2.allowed is True
