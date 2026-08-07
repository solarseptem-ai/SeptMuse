#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""LRU 嵌入缓存 — 避免相同文本重复计算嵌入向量。

透明包装 Embedder, 对调用方无感知。query 重复搜索时命中缓存, 跳过模型推理。
对 OnnxEmbedder (~50ms/query) 效果显著; HashEmbedder (<1ms) 开销可忽略。

线程安全: 内部加 threading.Lock, 支持 async (to_thread) + sync 并发访问。
缓存隔离: 返回 list 浅拷贝, 防止调用方修改污染缓存。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from septmuse.embedders.base import Embedder
from septmuse.observability.collector import MetricsCollector


class CachedEmbedder(Embedder):
    """带 LRU 缓存的 Embedder 包装器 (线程安全 + 缓存隔离)。"""

    def __init__(self, inner: Embedder, maxsize: int = 256) -> None:
        self.backend_name = "cached"
        self._inner = inner
        self._cache: OrderedDict[tuple[str, str | None], list[float]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        cache_key = (text, memory_action)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self._hits += 1
                MetricsCollector.get().inc("embed_cache_hits_total")
                return list(cached)
            self._misses += 1
            MetricsCollector.get().inc("embed_cache_misses_total")

        vec = self._inner._embed(text, memory_action=memory_action)

        with self._lock:
            self._cache[cache_key] = vec
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return list(vec)

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[int] = []

        with self._lock:
            for i, text in enumerate(texts):
                cache_key = (text, memory_action)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    MetricsCollector.get().inc("embed_cache_hits_total")
                    results[i] = list(cached)
                else:
                    self._misses += 1
                    MetricsCollector.get().inc("embed_cache_misses_total")
                    to_embed.append(i)

        if to_embed:
            embed_texts = [texts[i] for i in to_embed]
            embed_results = self._inner._embed_batch(embed_texts, memory_action=memory_action)

            with self._lock:
                for idx, vec in zip(to_embed, embed_results, strict=True):
                    cache_key = (texts[idx], memory_action)
                    results[idx] = list(vec)
                    self._cache[cache_key] = vec
                    self._cache.move_to_end(cache_key)
                    if len(self._cache) > self._maxsize:
                        self._cache.popitem(last=False)

        assert all(r is not None for r in results), "embed_batch internal error: None entry"
        return results

    @property
    def stats(self) -> dict[str, int]:
        """缓存统计信息。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }
