"""memory_action 接口测试 — 验证 ABC 签名 + CachedEmbedder cache key 隔离 + 向后兼容。"""

from __future__ import annotations


class TestEmbedderABCInterface:
    def test_embed_accepts_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello", memory_action="add")
        assert len(vec) == 64

    def test_embed_accepts_none_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello", memory_action=None)
        assert len(vec) == 64

    def test_embed_backward_compatible_no_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello")
        assert len(vec) == 64

    def test_embed_batch_accepts_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vecs = emb.embed_batch(["hello", "world"], memory_action="search")
        assert len(vecs) == 2
        assert all(len(v) == 64 for v in vecs)

    def test_embed_batch_backward_compatible(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2


class TestCachedEmbedderMemoryAction:
    def test_cache_key_includes_memory_action(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed("hello", memory_action="add")
        cached.embed("hello", memory_action="search")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2

    def test_same_memory_action_hits_cache(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed("hello", memory_action="add")
        cached.embed("hello", memory_action="add")

        stats = cached.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_none_memory_action_cached_separately_from_add(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed("hello")
        cached.embed("hello", memory_action="add")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2

    def test_embed_batch_cache_key_includes_memory_action(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed_batch(["hello"], memory_action="add")
        cached.embed_batch(["hello"], memory_action="search")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2
