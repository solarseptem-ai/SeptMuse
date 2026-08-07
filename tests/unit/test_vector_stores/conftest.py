"""向量存储测试 conftest。"""

import contextlib

import pytest


@pytest.fixture
def qdrant_path(tmp_path):
    """临时 Qdrant 本地嵌入路径。"""
    return str(tmp_path / "qdrant_test")


@pytest.fixture
def qdrant_store(qdrant_path):
    """临时 QdrantVectorStore（512 dim，本地嵌入，BM25 关闭）。"""
    from septmuse.storage.vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore(
        collection_name="test",
        embedding_model_dims=512,
        path=qdrant_path,
        enable_bm25=False,
    )
    yield store
    with contextlib.suppress(Exception):
        store.delete_collection()
    store.close()
