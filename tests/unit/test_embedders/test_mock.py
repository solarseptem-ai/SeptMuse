"""MockEmbedder 测试 — 固定向量验证。"""

from __future__ import annotations


class TestMockEmbedder:
    def test_inherits_embedder_abc(self):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert isinstance(emb, Embedder)

    def test_dimension_is_10(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert emb.dimension == 10

    def test_embed_returns_fixed_vector(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vec = emb.embed("anything")
        assert vec == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def test_embed_same_for_different_text(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert emb.embed("hello") == emb.embed("world")

    def test_embed_batch(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vecs = emb.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(v == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for v in vecs)

    def test_embed_accepts_memory_action(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vec = emb.embed("hello", memory_action="add")
        assert len(vec) == 10
