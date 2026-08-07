"""QdrantVectorStore CRUD + 本地路径 + collection 管理测试。"""


def test_insert_and_search(qdrant_store):
    """插入 + 搜索基础流程。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    results = qdrant_store.search_vectors([0.1] * 512, top_k=1)
    assert len(results) == 1
    assert results[0].id == "m1"


def test_delete(qdrant_store):
    """插入后删除。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    assert qdrant_store.delete_vector("m1") is True
    assert qdrant_store.get_vector("m1") is None


def test_get_vector(qdrant_store):
    """插入后取单条向量。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    entry = qdrant_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert entry.payload["user_id"] == "alice"


def test_update_vector(qdrant_store):
    """插入后更新向量和 payload。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    assert qdrant_store.update_vector("m1", [0.2] * 512, {"user_id": "bob"}) is True
    entry = qdrant_store.get_vector("m1")
    assert entry is not None
    assert entry.payload["user_id"] == "bob"


def test_list_collections(qdrant_store):
    """list_collections 应包含当前 collection。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    collections = qdrant_store.list_collections()
    assert "test" in collections


def test_get_collection_info(qdrant_store):
    """get_collection_info 返回 name 和 count。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    info = qdrant_store.get_collection_info()
    assert info["name"] == "test"
    assert info["count"] >= 1


def test_reset_collection(qdrant_store):
    """reset_collection 后 count 归零。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    qdrant_store.reset_collection()
    info = qdrant_store.get_collection_info()
    assert info["count"] == 0
