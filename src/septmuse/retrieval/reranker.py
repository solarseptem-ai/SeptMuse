#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""兼容层 — reranker 已迁移到 septmuse.rerankers/ 目录。

本文件保留旧接口 (rerank 操作 list[HybridResult]) 供向后兼容。
新代码应直接使用 septmuse.rerankers。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from septmuse.rerankers.base import BaseReranker
from septmuse.rerankers.cross_encoder import CrossEncoderReranker as _CEInner
from septmuse.rerankers.llm import LLMReranker as _LLMInner
from septmuse.rerankers.mmr import MMRReranker as _MMRInner
from septmuse.rerankers.noop import NoopReranker as _NoopInner
from septmuse.rerankers.sentence_transformer import SentenceTransformerReranker as _STInner

if TYPE_CHECKING:
    pass

# 旧别名 (测试引用 Reranker ABC)
Reranker = BaseReranker


class NoopReranker(_NoopInner):
    """旧接口兼容: rerank(query, list[HybridResult]) -> list[HybridResult]。"""
    def rerank(self, query, results, *, top_k=None, search_filter=None):
        # 透传: 不改变 score, 只截断 top_k
        if top_k is not None:
            return results[:top_k]
        return results


class MMRReranker(_MMRInner):
    """旧接口兼容。"""
    def rerank(self, query, results, *, top_k=None, search_filter=None):
        if not results:
            return []
        from septmuse.rerankers.strategies.full_memory import FullMemoryStrategy
        strategy = FullMemoryStrategy()
        tracker, documents = strategy.prepare(results)
        tk = top_k or len(results)
        scored = super().rerank(query, documents, top_k=tk)
        return strategy.reconstruct(scored, tracker, results, tk, search_filter)


class CrossEncoderReranker(_CEInner):
    """旧接口兼容。"""
    def rerank(self, query, results, *, top_k=None, search_filter=None):
        if not results:
            return []
        from septmuse.rerankers.strategies.full_memory import FullMemoryStrategy
        strategy = FullMemoryStrategy()
        tracker, documents = strategy.prepare(results)
        tk = top_k or len(results)
        scored = super().rerank(query, documents, top_k=tk)
        return strategy.reconstruct(scored, tracker, results, tk, search_filter)


class LLMReranker(_LLMInner):
    """旧接口兼容。"""
    def rerank(self, query, results, *, top_k=None, search_filter=None):
        if not results:
            return []
        from septmuse.rerankers.strategies.full_memory import FullMemoryStrategy
        strategy = FullMemoryStrategy()
        tracker, documents = strategy.prepare(results)
        tk = top_k or len(results)
        scored = super().rerank(query, documents, top_k=tk)
        return strategy.reconstruct(scored, tracker, results, tk, search_filter)


class SentenceTransformerReranker(_STInner):
    """旧接口兼容。"""
    def rerank(self, query, results, *, top_k=None, search_filter=None):
        if not results:
            return []
        from septmuse.rerankers.strategies.full_memory import FullMemoryStrategy
        strategy = FullMemoryStrategy()
        tracker, documents = strategy.prepare(results)
        tk = top_k or len(results)
        scored = super().rerank(query, documents, top_k=tk)
        return strategy.reconstruct(scored, tracker, results, tk, search_filter)


def _resolve_reranker(backend="noop", *, embedder=None, llm=None, model_cache_dir=None, api_key=None):
    """旧工厂函数 (兼容)。"""
    if backend == "noop":
        return NoopReranker()
    if backend == "mmr":
        if embedder is None:
            raise ValueError("MMRReranker requires an embedder")
        return MMRReranker(embedder)
    if backend == "cross_encoder":
        return CrossEncoderReranker(model_cache_dir=model_cache_dir)
    if backend == "llm":
        if llm is None:
            raise ValueError("LLMReranker requires an LLM instance")
        return LLMReranker(llm=llm)
    if backend == "sentence_transformer":
        return SentenceTransformerReranker()
    from septmuse.rerankers import create_reranker
    return create_reranker(backend, embedder=embedder, llm=llm, model_cache_dir=model_cache_dir, api_key=api_key)
