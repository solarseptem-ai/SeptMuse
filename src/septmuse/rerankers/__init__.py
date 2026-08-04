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
"""Reranker 模块 — 6 种重排器 + 策略层。

架构:
- reranker 管"用什么打分" (noop/mmr/cross_encoder/llm/batch_llm/cohere)
- strategy 管"打什么内容" (full_memory/single_turn)
- 两者正交组合: reranker 操作 list[str], strategy 做 HybridResult <-> str 转换

用法:
    from septmuse.rerankers import create_reranker
    from septmuse.rerankers.strategies import RerankerStrategyFactory

    reranker = create_reranker("cross_encoder")
    strategy = RerankerStrategyFactory.create("full_memory")
    tracker, documents = strategy.prepare(results)
    scored = reranker.rerank(query, documents, top_k=5)
    results = strategy.reconstruct(scored, tracker, results, top_k=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from septmuse.rerankers.base import BaseReranker
from septmuse.rerankers.batch_llm import BatchLLMReranker
from septmuse.rerankers.cohere import CohereReranker
from septmuse.rerankers.cross_encoder import CrossEncoderReranker
from septmuse.rerankers.llm import LLMReranker
from septmuse.rerankers.mmr import MMRReranker
from septmuse.rerankers.noop import NoopReranker

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder
    from septmuse.llms.base import LLM

__all__ = [
    "BaseReranker",
    "BatchLLMReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "MMRReranker",
    "NoopReranker",
    "create_reranker",
]


def create_reranker(
    backend: str = "noop",
    *,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    model_cache_dir: str | None = None,
    api_key: str | None = None,
) -> BaseReranker:
    """工厂函数: 按 backend 名称创建 Reranker 实例。

    支持: noop / mmr / cross_encoder / llm / batch_llm / cohere。
    """
    from septmuse.services.providers import reranker_provider

    if backend == "mmr" and embedder is None:
        raise ValueError("MMRReranker requires an embedder")
    if backend in ("llm", "batch_llm") and llm is None:
        raise ValueError(f"{backend} reranker requires an LLM instance")
    try:
        return reranker_provider.resolve(
            backend, embedder=embedder, llm=llm, model_cache_dir=model_cache_dir, api_key=api_key
        )
    except ValueError as e:
        if "Unknown backend" in str(e):
            raise ValueError(f"Unknown reranker backend: {backend}") from e
        raise
