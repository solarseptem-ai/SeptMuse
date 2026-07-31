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
"""Reranker 框架 (借鉴 mem0 BaseReranker + graphiti CrossEncoderClient)。

后处理重排模式: HybridRetriever.search() → Reranker.rerank() → 返回。
Reranker 操作 list[HybridResult], 保留原始字段, 更新 score。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger
from septmuse.retrieval.hybrid import HybridResult

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder
    from septmuse.llms.base import LLM

logger = get_logger(__name__)


class Reranker(ABC):
    """重排器抽象基类 (借鉴 mem0 BaseReranker + graphiti CrossEncoderClient)。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        """对检索结果重排, 返回按相关性降序排列的 HybridResult 列表。

        实现方应:
        - 保留原始 HybridResult 的其他字段 (id, memory, metadata, created_at)
        - 更新 score 字段为重排后的分数
        """
        ...


class NoopReranker(Reranker):
    """透传 reranker, 不改变顺序和 score (借鉴 MemOS NoopReranker)。"""

    def __init__(self, **kwargs) -> None:
        pass

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        if top_k is not None:
            return results[:top_k]
        return results


class MMRReranker(Reranker):
    """最大边际相关性 reranker (借鉴 graphiti maximal_marginal_relevance)。

    贪心迭代选择: 每轮从未选集合中选 MMR 分数最高的候选加入 selected。
    mmr = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))
    去冗余: 相似度 >0.9 的结果只保留排名靠前的一个。
    """

    def __init__(self, embedder: Embedder, lambda_param: float = 0.7, **kwargs) -> None:
        self.embedder = embedder
        self.lambda_param = lambda_param

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        if not results:
            return []

        tk = top_k or len(results)
        query_emb = self.embedder.embed(query)

        doc_embs: list[list[float]] = []
        for r in results:
            doc_embs.append(self.embedder.embed(r.memory))

        query_sims = [self._cosine(query_emb, de) for de in doc_embs]

        selected: list[int] = []
        remaining = list(range(len(results)))

        while remaining and len(selected) < tk:
            best_idx = -1
            best_score = -float("inf")
            for i in remaining:
                max_sim = max(self._cosine(doc_embs[i], doc_embs[j]) for j in selected) if selected else 0.0
                mmr = self.lambda_param * query_sims[i] - (1 - self.lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

            # 去冗余: 相似度 >0.9 的剩余候选跳过
            to_remove = []
            for j in remaining:
                if self._cosine(doc_embs[best_idx], doc_embs[j]) > 0.9:
                    to_remove.append(j)
            for j in to_remove:
                remaining.remove(j)

        out = [results[i] for i in selected]
        for rank, r in enumerate(out):
            r.score = query_sims[selected[rank]]
        return out


class CrossEncoderReranker(Reranker):
    """ONNX cross-encoder reranker (借鉴 graphiti BGERerankerClient + mem0 TS CrossEncoderReranker)。

    延迟 import onnxruntime, 不可用时降级为 Noop + 日志警告。
    模型: BAAI/bge-reranker-v2-m3 ONNX 量化版, ModelScope 下载。
    sigmoid(logit) 归一化到 [0,1]。
    """

    def __init__(self, model_cache_dir: str | None = None, **kwargs) -> None:
        self._model_cache_dir = model_cache_dir
        self._session = None
        self._degraded = False
        self._init_attempted = False

    def _init_model(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import onnxruntime as ort

            self._session = ort
        except ImportError:
            logger.warning("cross_encoder_reranker_degraded", reason="onnxruntime not installed")
            self._degraded = True

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        if not results:
            return []

        self._init_model()

        if self._degraded or self._session is None:
            if top_k is not None:
                return results[:top_k]
            return results

        # 实际模型推理待 P3/P4 补 (modelscope 下载 + onnxruntime session)
        logger.info("cross_encoder_reranker_not_implemented", reason="model loading deferred")
        if top_k is not None:
            return results[:top_k]
        return results


class LLMReranker(Reranker):
    """LLM 打分 reranker (借鉴 mem0 LLMReranker)。

    构造时传入 LLM 实例, LLM.complete() 逐条打分 0-1。
    _extract_score 正则提取数字, clamp [0,1], 无数字返回 0.5。
    无 LLM 实例时抛 ValueError。
    """

    _MAX_INPUT_LEN = 4000

    _SYSTEM_PROMPT = (
        "You are a relevance scoring assistant. "
        "Given a query and a document, score how relevant the document is to the query.\n\n"
        "Score the relevance on a scale from 0.0 to 1.0, where:\n"
        "- 1.0 = Perfectly relevant\n"
        "- 0.0 = Not relevant\n\n"
        "Respond with only a single numerical score between 0.0 and 1.0. "
        "Do not include any explanation."
    )

    def __init__(self, llm: LLM | None = None, **kwargs) -> None:
        self._llm = llm

    def _extract_score(self, response_text: str) -> float:
        matches = re.findall(r"-?\d+\.\d+", response_text) or re.findall(r"-?\d+", response_text)
        if matches:
            score = float(matches[0])
            return min(max(score, 0.0), 1.0)
        return 0.5

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        if not results:
            return []

        if self._llm is None:
            raise ValueError("LLMReranker requires an LLM instance")

        scored: list[HybridResult] = []
        for r in results:
            safe_query = query[: self._MAX_INPUT_LEN]
            safe_doc = r.memory[: self._MAX_INPUT_LEN]
            user_prompt = f"Query: {safe_query}\n\nDocument: {safe_doc}"
            try:
                response = self._llm.complete(self._SYSTEM_PROMPT, user_prompt)
                score = self._extract_score(response)
            except Exception as e:
                logger.warning("llm_rerank_failed", memory_id=r.id, error=str(e))
                score = 0.5
            scored.append(
                HybridResult(
                    id=r.id,
                    memory=r.memory,
                    score=score,
                    vector_score=r.vector_score,
                    bm25_score=r.bm25_score,
                    entity_boost=r.entity_boost,
                    metadata=r.metadata,
                    created_at=r.created_at,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored


def _resolve_reranker(
    backend: str = "noop",
    *,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    model_cache_dir: str | None = None,
) -> Reranker:
    """工厂函数: 根据 backend 字符串创建 Reranker 实例 (通过 ServiceProvider)。"""
    from septmuse.services.providers import reranker_provider

    if backend == "mmr" and embedder is None:
        raise ValueError("MMRReranker requires an embedder")
    return reranker_provider.resolve(
        backend, embedder=embedder, llm=llm, model_cache_dir=model_cache_dir
    )
