"""QdrantVectorStore 过滤操作符测试。"""

_DATA = [
    {"user_id": "alice", "score": 5, "tags": "work"},
    {"user_id": "alice", "score": 10, "tags": "personal"},
    {"user_id": "bob", "score": 15, "tags": "work"},
]


def _seed(qdrant_store):
    qdrant_store.insert_vectors(
        [[0.1] * 512, [0.2] * 512, [0.3] * 512],
        ["m1", "m2", "m3"],
        _DATA,
    )


def test_filter_exact(qdrant_store):
    """简单精确匹配: {"user_id": "alice"}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=10, filters={"user_id": "alice"})
    assert {r.id for r in results} == {"m1", "m2"}


def test_filter_eq(qdrant_store):
    """eq 操作符: {"user_id": {"eq": "bob"}}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=10, filters={"user_id": {"eq": "bob"}})
    assert len(results) == 1
    assert results[0].id == "m3"


def test_filter_ne(qdrant_store):
    """ne 操作符: {"user_id": {"ne": "bob"}}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=10, filters={"user_id": {"ne": "bob"}})
    assert {r.id for r in results} == {"m1", "m2"}


def test_filter_gte(qdrant_store):
    """gte 范围操作符: {"score": {"gte": 10}}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=10, filters={"score": {"gte": 10}})
    assert {r.id for r in results} == {"m2", "m3"}


def test_filter_in(qdrant_store):
    """in 操作符: {"user_id": {"in": ["alice", "bob"]}}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=10, filters={"user_id": {"in": ["alice", "bob"]}})
    assert len(results) == 3


def test_filter_and(qdrant_store):
    """AND 逻辑组合: {"AND": [{"user_id": "alice"}, {"score": {"gte": 10}}]}。"""
    _seed(qdrant_store)
    results = qdrant_store.search_vectors(
        [0.1] * 512,
        top_k=10,
        filters={"AND": [{"user_id": "alice"}, {"score": {"gte": 10}}]},
    )
    assert len(results) == 1
    assert results[0].id == "m2"
