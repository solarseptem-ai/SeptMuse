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
"""混合检索 — BM25 关键词 + 向量语义 RRF 融合 (架构文档 §5.2, 借鉴 mem0 + ReMe)。

借鉴:
- mem0 scoring.py: BM25 归一化 + 加性融合 (本模块用 RRF 替代, 更简单稳健)
- ReMe search.py _rrf_merge: RRF (Reciprocal Rank Fusion) 融合向量+关键词
  fused_score = vector_weight/(k+vector_rank) + keyword_weight/(k+keyword_rank)

BM25 实现: 纯 Python 无外部依赖 (对齐 mem0 utils/scoring.py 自实现 BM25)。
k1=1.5, b=0.75 (标准 BM25 参数)。

详见 docs/specs/agent-memory-architecture.md §5.2 检索策略。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.storage.base import MemoryStore

if TYPE_CHECKING:
    from septmuse.extraction.entity import EntityExtractor
    from septmuse.storage.entity_store import EntityStore

logger = get_logger(__name__)

# RRF 常数 k=60 (对齐 ReMe search.py _rrf_merge)
RRF_K = 60

# BM25 标准参数
BM25_K1 = 1.5
BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """简单分词 (小写化 + 非字母数字分割, 对齐 mem0 BM25 预处理)。"""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w]


class BM25Scorer:
    """BM25 关键词评分器 (纯 Python, 无外部依赖)。

    用法:
        scorer = BM25Scorer()
        scorer.index(["hello world", "foo bar baz"])
        scores = scorer.score("hello")  # [score_for_doc0, score_for_doc1]
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}  # 词 → 包含该词的文档数
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._n: int = 0

    def index(self, documents: list[str]) -> None:
        """建立 BM25 索引。"""
        self._docs = [_tokenize(doc) for doc in documents]
        self._n = len(self._docs)
        self._doc_len = [len(doc) for doc in self._docs]
        self._avgdl = sum(self._doc_len) / self._n if self._n > 0 else 0.0
        self._doc_freq = {}
        for doc in self._docs:
            seen: set[str] = set()
            for word in doc:
                if word not in seen:
                    self._doc_freq[word] = self._doc_freq.get(word, 0) + 1
                    seen.add(word)
        logger.debug("bm25_indexed", docs=self._n, avgdl=self._avgdl, vocab=len(self._doc_freq))

    def score(self, query: str) -> list[float]:
        """对每篇文档计算 BM25 分数。"""
        query_terms = _tokenize(query)
        if not query_terms or self._n == 0:
            return [0.0] * self._n

        scores = [0.0] * self._n
        for term in query_terms:
            n_q = self._doc_freq.get(term, 0)
            if n_q == 0:
                continue
            idf = math.log((self._n - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for i, doc in enumerate(self._docs):
                f_qd = doc.count(term)
                if f_qd == 0:
                    continue
                dl = self._doc_len[i]
                denom = f_qd + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl) if self._avgdl > 0 else f_qd
                scores[i] += idf * (f_qd * (self.k1 + 1.0)) / denom
        return scores


@dataclass
class HybridResult:
    """混合检索结果项。"""

    id: str
    memory: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    entity_boost: float = 0.0
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


class HybridRetriever:
    """混合检索器 (BM25 + 向量 RRF 融合, 对齐 ReMe _rrf_merge)。

    用法:
        retriever = HybridRetriever(store, embedder)
        results = retriever.search("alice python", user_id="alice")
        # results 含 RRF 融合后的排序结果
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        entity_extractor: EntityExtractor | None = None,
        entity_store: EntityStore | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.entity_extractor = entity_extractor
        self.entity_store = entity_store
        if entity_extractor is not None and entity_store is None:
            logger.warning("entity_boost_disabled", reason="entity_store missing but extractor present")
        if entity_store is not None and entity_extractor is None:
            logger.warning("entity_boost_disabled", reason="entity_extractor missing but store present")

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        explain: bool = False,
    ) -> list[HybridResult]:
        """BM25 + 向量 RRF 融合检索 (对齐 ReMe _rrf_merge)。

        session_id: 仅搜该会话的记忆 (None=不限, 对齐 mem0 run_id)。
        """
        # 1. 获取全部候选 (over-fetch)
        all_memories = self.store.get_all(user_id=user_id, session_id=session_id)
        if not all_memories:
            return []

        documents = [m["memory"] for m in all_memories]
        ids = [m["id"] for m in all_memories]
        metadatas = [m.get("metadata", {}) for m in all_memories]
        created_ats = [m.get("created_at") for m in all_memories]

        # 2. 向量检索 (Layer 1: 语义召回)
        emb = self.embedder.embed(query)
        vector_results = self.store.search(emb, user_id=user_id, top_k=len(all_memories), threshold=threshold)
        vector_rank: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        for rank, r in enumerate(vector_results):
            vector_rank[r["id"]] = rank
            vector_scores[r["id"]] = r["score"]

        # 3. BM25 关键词检索 (Layer 2: 词频召回)
        bm25 = BM25Scorer()
        bm25.index(documents)
        bm25_scores = bm25.score(query)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        keyword_rank: dict[str, int] = {}
        for rank, i in enumerate(bm25_ranked):
            keyword_rank[ids[i]] = rank

        # 3.5 Entity boost (第三信号, 借鉴 mem0 _search_vector_store scoring)
        entity_boosts: dict[str, float] = {}
        if self.entity_extractor is not None and self.entity_store is not None:
            try:
                entities = self.entity_extractor.extract(query)
                for entity in entities:
                    matches = self.entity_store.search(entity.text, user_id=user_id, top_k=10)
                    for match in matches:
                        linked_ids = match.get("linked_memory_ids", [])
                        n = len(linked_ids)
                        boost = 0.5 * 1.0 / (1.0 + 0.001 * (n - 1) ** 2) if n > 0 else 0.0
                        for eid in linked_ids:
                            entity_boosts[eid] = entity_boosts.get(eid, 0.0) + boost
            except Exception as e:
                logger.warning("entity_boost_failed", error=str(e))

        # 4. RRF 融合 (对齐 ReMe _rrf_merge) + entity boost (第三信号加性融合)
        results: list[HybridResult] = []
        for i, mid in enumerate(ids):
            v_rank = vector_rank.get(mid)
            k_rank = keyword_rank.get(mid)
            fused = 0.0
            v_score = vector_scores.get(mid, 0.0)
            if v_rank is not None:
                fused += self.vector_weight / (RRF_K + v_rank + 1)
            if k_rank is not None and bm25_scores[i] > 0:
                fused += self.keyword_weight / (RRF_K + k_rank + 1)
            e_boost = entity_boosts.get(mid, 0.0)
            fused += e_boost
            if fused > 0:
                meta = dict(metadatas[i]) if metadatas[i] is not None else {}
                if explain:
                    meta["score_details"] = {
                        "vector": v_score,
                        "bm25": bm25_scores[i],
                        "entity_boost": e_boost,
                        "combined": fused,
                    }
                results.append(
                    HybridResult(
                        id=mid,
                        memory=documents[i],
                        score=fused,
                        vector_score=v_score,
                        bm25_score=bm25_scores[i],
                        entity_boost=e_boost,
                        metadata=meta,
                        created_at=created_ats[i],
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            "hybrid_search_done",
            user_id=user_id,
            candidates=len(all_memories),
            vector_hits=len(vector_results),
            bm25_hits=sum(1 for s in bm25_scores if s > 0),
            returned=len(results[:top_k]),
        )
        return results[:top_k]
