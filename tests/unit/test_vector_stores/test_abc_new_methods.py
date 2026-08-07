"""VectorStoreBase 新方法测试（用 SQLAlchemyVectorStore 验证 fallback）。"""

from sqlalchemy import create_engine

from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore


def test_update_vector():
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors([[0.1] * 4], ["m1"], [{"user_id": "a"}])
    assert store.update_vector("m1", [0.2] * 4, {"user_id": "b"}) is True


def test_delete_collection():
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors([[0.1] * 4], ["m1"], [{"user_id": "a"}])
    store.delete_collection()
    info = store.get_collection_info()
    assert info["count"] == 0


def test_keyword_search_returns_none():
    """SQLAlchemy 后端 keyword_search 默认返回 None。"""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    assert store.keyword_search("test") is None


def test_search_batch_default():
    """search_batch 默认循环 search_vectors。"""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors(
        [[0.1] * 4, [0.2] * 4],
        ["m1", "m2"],
        [{"user_id": "a"}, {"user_id": "a"}],
    )
    results = store.search_batch(["q1", "q2"], [[0.1] * 4, [0.2] * 4], top_k=2)
    assert len(results) == 2
