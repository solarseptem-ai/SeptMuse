"""Qdrant BM25 稀疏向量测试。fastembed 未装时 skip。"""

import contextlib

import pytest

try:
    import fastembed  # noqa: F401

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


@pytest.fixture
def bm25_store(tmp_path):
    from septmuse.storage.vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore(
        collection_name="test_bm25",
        embedding_model_dims=512,
        path=str(tmp_path / "qdrant_bm25"),
        enable_bm25=True,
    )
    yield store
    with contextlib.suppress(Exception):
        store.delete_collection()
    store.close()


@pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not installed")
def test_bm25_keyword_search(bm25_store):
    """BM25 关键词搜索。"""
    bm25_store.insert_vectors(
        [[0.1] * 512, [0.2] * 512],
        ["m1", "m2"],
        [
            {"user_id": "alice", "data": "I love programming in Python"},
            {"user_id": "alice", "data": "The weather is nice today"},
        ],
    )
    results = bm25_store.keyword_search("programming", top_k=2)
    assert results is not None
    assert len(results) > 0


def test_bm25_returns_none_when_disabled(qdrant_store):
    """enable_bm25=False 时 keyword_search 返回 None。"""
    assert qdrant_store.keyword_search("test") is None
