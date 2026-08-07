"""Qdrant search_batch 测试."""


def test_search_batch(qdrant_store):
    qdrant_store.insert_vectors(
        [[0.1] * 512, [0.2] * 512],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    results = qdrant_store.search_batch(
        ["q1", "q2"],
        [[0.1] * 512, [0.2] * 512],
        top_k=2,
    )
    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
