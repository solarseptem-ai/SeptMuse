"""Embedder 模板方法测试 — embed() 调用后 embed_duration_seconds 记录。"""

from prometheus_client import REGISTRY

from septmuse.embedders.cached import CachedEmbedder
from septmuse.embedders.hash import HashEmbedder
from septmuse.observability.collector import MetricsCollector


def test_embed_records_duration():
    """embed() 调用后 embed_duration_seconds 被记录。"""
    MetricsCollector.get().configure(enabled=True)
    embedder = HashEmbedder()
    result = embedder.embed("hello")
    assert len(result) > 0

    samples = list(REGISTRY.collect())
    emb_metric = [s for s in samples if s.name == "septmuse_embed_duration_seconds"]
    assert len(emb_metric) > 0
    found = any(s.labels.get("backend") == "hash" for s in emb_metric[0].samples)
    assert found


def test_embed_batch_records_duration():
    """embed_batch() 调用后 embed_batch_duration_seconds 被记录。"""
    MetricsCollector.get().configure(enabled=True)
    embedder = HashEmbedder()
    results = embedder.embed_batch(["hello", "world"])
    assert len(results) == 2

    samples = list(REGISTRY.collect())
    batch_metric = [s for s in samples if s.name == "septmuse_embed_batch_duration_seconds"]
    assert len(batch_metric) > 0
    found = any(s.labels.get("backend") == "hash" for s in batch_metric[0].samples)
    assert found


def test_cached_embedder_cache_counters():
    """CachedEmbedder cache hit/miss 计数。"""
    MetricsCollector.get().configure(enabled=True)
    inner = HashEmbedder()
    cached = CachedEmbedder(inner)
    cached.embed("hello")
    cached.embed("hello")

    samples = list(REGISTRY.collect())
    # prometheus_client 会自动去掉 Counter 名的 _total 后缀
    hits = [s for s in samples if s.name == "septmuse_embed_cache_hits"]
    misses = [s for s in samples if s.name == "septmuse_embed_cache_misses"]
    assert len(hits) > 0
    assert len(misses) > 0
    hit_total = sum(s.value for s in hits[0].samples)
    miss_total = sum(s.value for s in misses[0].samples)
    assert hit_total >= 1.0
    assert miss_total >= 1.0


def test_noop_when_not_configured():
    """未 configure 时 embed() 仍然正常工作。"""
    embedder = HashEmbedder()
    result = embedder.embed("hello")
    assert len(result) > 0
