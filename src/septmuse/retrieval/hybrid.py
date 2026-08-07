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
"""混合检索 — BM25 关键词 + 向量语义 RRF 融合 (架构文档 §5.2)。

BM25 归一化 + RRF (Reciprocal Rank Fusion) 融合向量+关键词:
fused_score = vector_weight/(k+vector_rank) + keyword_weight/(k+keyword_rank)

BM25 实现: 纯 Python 无外部依赖, k1=1.5, b=0.75 (标准 BM25 参数)。

详见 docs/specs/agent-memory-architecture.md §5.2 检索策略。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger
from septmuse.core.tokenizer import tokenize
from septmuse.embedders.base import Embedder
from septmuse.observability.hooks import time_block
from septmuse.storage.base import MemoryStore

if TYPE_CHECKING:
    from septmuse.extraction.entity import EntityExtractor
    from septmuse.storage.relational_stores.entity_store import EntityStore

logger = get_logger(__name__)

# RRF 常数 k=60 (标准参数)
RRF_K = 60

# BM25 标准参数
BM25_K1 = 1.5
BM25_B = 0.75

# 三路并发检索超时 (秒), 超时降级空结果
SEARCH_TIMEOUT = 5.0

# 查询结果缓存 TTL (秒), 记忆变更后自动失效
SEARCH_CACHE_TTL = 300.0


def _await_future(fut: Future, name: str, default: Any) -> Any:
    """等待 future 完成, 超时或异常时返回默认值。"""
    try:
        return fut.result(timeout=SEARCH_TIMEOUT)
    except FuturesTimeoutError:
        logger.warning(f"{name}_search_timeout", timeout=SEARCH_TIMEOUT)
        return default
    except Exception as e:
        logger.warning(f"{name}_search_error", error=str(e))
        return default


# 简单英文后缀列表 (短 → 长, 先匹配短后缀: likes→like 而非 lik)
_SUFFIXES = ("s", "es", "ed", "ing", "ies", "ied")


def _lemmatize_word(word: str) -> str:
    """简单词形还原: 英文去后缀, 中文保持不变 (对齐 mem0 lemmatize_for_bm25)。"""
    if any("\u4e00" <= c <= "\u9fff" for c in word):
        return word
    if len(word) > 3:
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
    return word


def lemmatize_for_bm25(text: str) -> str:
    """BM25 词形还原预处理 (英文去后缀, 中文不变)。

    >>> lemmatize_for_bm25("I love running and coding")
    'i love run and code'
    >>> lemmatize_for_bm25("我喜欢编程")
    '我 喜 欢 编 程'
    """
    tokens = tokenize(text)
    return " ".join(_lemmatize_word(t) for t in tokens)


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
        """建立 BM25 索引 (含词形还原)。"""
        self._docs = [[_lemmatize_word(w) for w in tokenize(doc)] for doc in documents]
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
        """对每篇文档计算 BM25 分数 (含词形还原)。"""
        query_terms = [_lemmatize_word(w) for w in tokenize(query)]
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
    """混合检索器 (BM25 + 向量 RRF 融合)。

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
        cache_ttl: float = SEARCH_CACHE_TTL,
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
        # 查询结果缓存 (TTL + thread-safe)
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, list[HybridResult]]] = {}
        self._cache_lock = threading.Lock()

    def _cache_key(
        self, query: str, user_id: str, session_id: str | None, top_k: int,
        threshold: float, filters: dict[str, Any] | None, explain: bool,
    ) -> str:
        """构建缓存 key: query_hash:user_id:session_id:top_k:threshold:filters_hash:explain。"""
        qhash = hashlib.md5(query.encode()).hexdigest()
        fhash = hashlib.md5(
            json.dumps(filters or {}, sort_keys=True, default=str).encode()
        ).hexdigest()
        sid = session_id or ""
        return f"{qhash}:{user_id}:{sid}:{top_k}:{threshold}:{fhash}:{explain}"

    def invalidate_cache(self, user_id: str | None = None) -> None:
        """清除查询缓存。user_id=None 清所有, 指定 user_id 只清该用户的。"""
        with self._cache_lock:
            if user_id is None:
                self._cache.clear()
            else:
                keys_to_del = [k for k in self._cache if f":{user_id}:" in k]
                for k in keys_to_del:
                    del self._cache[k]

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        explain: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """BM25 + 向量 RRF 融合检索。

        session_id: 仅搜该会话的记忆 (None=不限)。
        filters: 字段过滤字典 (如 {"session_id":"s1"}), None=不过滤。

        优化: over-fetch (internal_limit = max(top_k*4, 60)) 替代全量 get_all,
        BM25 仅在向量召回的候选池上索引, 避免大记忆库全量加载。

        缓存: 相同 query+params 5 分钟内返回缓存结果 (TTL 300s),
        Memory.add/update/delete 时调 invalidate_cache 清除。
        """
        # 缓存检查 (命中则直接返回, 跳过三路并发检索)
        cache_key = self._cache_key(query, user_id, session_id, top_k, threshold, filters, explain)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                expiry, cached_results = cached
                if time.time() < expiry:
                    logger.info("hybrid_search_cache_hit", user_id=user_id, cached=len(cached_results))
                    return cached_results
                del self._cache[cache_key]

        # 三路并发检索: 向量 + BM25 + entity boost (延迟 = max(v,b,e) 而非 v+b+e)
        internal_limit = max(top_k * 4, 60)

        def _vector_path() -> list[dict[str, Any]]:
            with time_block("hybrid_search_components_seconds", {"component": "vector"}):
                emb = self.embedder.embed(query)
                return self.store.search(
                    emb,
                    user_id=user_id,
                    session_id=session_id,
                    top_k=internal_limit,
                    threshold=threshold,
                    filters=filters,
                )

        # VectorStore 内置 BM25 过滤条件 (user_id 隔离 + session_id)
        vs_filters: dict[str, Any] = {"user_id": user_id}
        if session_id is not None:
            vs_filters["session_id"] = session_id

        def _keyword_path() -> list[dict[str, Any]] | None:
            with time_block("hybrid_search_components_seconds", {"component": "keyword"}):
                # 优先用 VectorStore 内置 keyword_search (Qdrant BM25 sparse)
                vs = getattr(self.store, "_vector_store", None)
                if vs is not None and hasattr(vs, "keyword_search"):
                    vs_results = vs.keyword_search(query, top_k=internal_limit, filters=vs_filters)
                    if vs_results is not None:
                        return [
                            {
                                "id": r.id,
                                "score": r.score,
                                "memory": (r.payload or {}).get("data", (r.payload or {}).get("text", "")),
                            }
                            for r in vs_results
                        ]
                # 回退到外部 KeywordIndexBase (SQLite BM25 / rank_bm25)
                try:
                    return self.store.keyword_search(
                        query, user_id=user_id, session_id=session_id, top_k=internal_limit
                    )
                except Exception:
                    return None  # 降级标记, 用候选集 BM25

        def _entity_path() -> dict[str, float]:
            with time_block("hybrid_search_components_seconds", {"component": "entity"}):
                if self.entity_extractor is None or self.entity_store is None:
                    return {}
                try:
                    entities = self.entity_extractor.extract(query)
                    boosts: dict[str, float] = {}
                    for entity in entities:
                        matches = self.entity_store.search(entity.text, user_id=user_id, top_k=10)
                        for match in matches:
                            linked_ids = match.get("linked_memory_ids", [])
                            n = len(linked_ids)
                            boost = 0.5 / (RRF_K + 1) if n > 0 else 0.0
                            for eid in linked_ids:
                                boosts[eid] = boosts.get(eid, 0.0) + boost
                    return boosts
                except Exception as e:
                    logger.warning("entity_boost_failed", error=str(e))
                    return {}

        executor = ThreadPoolExecutor(max_workers=3)
        try:
            future_v = executor.submit(_vector_path)
            future_k = executor.submit(_keyword_path)
            future_e = executor.submit(_entity_path)

            vector_results = _await_future(future_v, "vector", [])
            kw_results = _await_future(future_k, "keyword", None)
            entity_boosts = _await_future(future_e, "entity", {})
        finally:
            executor.shutdown(wait=False)

        if not vector_results:
            return []

        # 候选集构建 (从向量结果)
        documents = [r["memory"] for r in vector_results]
        ids = [r["id"] for r in vector_results]
        metadatas = [r.get("metadata", {}) for r in vector_results]
        created_ats = [r.get("created_at") for r in vector_results]
        vector_rank: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        for rank, r in enumerate(vector_results):
            vector_rank[r["id"]] = rank
            vector_scores[r["id"]] = r["score"]

        # BM25 检索结果处理 (正常路径 or 降级到候选集 BM25)
        bm25_scores: dict[str, float] = {}
        keyword_rank: dict[str, int] = {}
        if kw_results is not None:
            for rank, r in enumerate(kw_results):
                mid = r["id"]
                keyword_rank[mid] = rank
                bm25_scores[mid] = r.get("score", 0.0)
        else:
            bm25 = BM25Scorer()
            bm25.index(documents)
            scores = bm25.score(query)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(ranked):
                keyword_rank[ids[i]] = rank
                bm25_scores[ids[i]] = scores[i]

        # 4. RRF 融合 + entity boost (第三信号加性融合)
        results: list[HybridResult] = []
        for i, mid in enumerate(ids):
            v_rank = vector_rank.get(mid)
            k_rank = keyword_rank.get(mid)
            fused = 0.0
            v_score = vector_scores.get(mid, 0.0)
            b_score = bm25_scores.get(mid, 0.0)
            if v_rank is not None:
                fused += self.vector_weight / (RRF_K + v_rank + 1)
            if k_rank is not None and b_score > 0:
                fused += self.keyword_weight / (RRF_K + k_rank + 1)
            e_boost = entity_boosts.get(mid, 0.0)
            fused += e_boost
            if fused > 0:
                meta = dict(metadatas[i]) if metadatas[i] is not None else {}
                if explain:
                    meta["score_details"] = {
                        "vector": v_score,
                        "bm25": b_score,
                        "entity_boost": e_boost,
                        "combined": fused,
                    }
                results.append(
                    HybridResult(
                        id=mid,
                        memory=documents[i],
                        score=fused,
                        vector_score=v_score,
                        bm25_score=b_score,
                        entity_boost=e_boost,
                        metadata=meta,
                        created_at=created_ats[i],
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        final = results[:top_k]

        # 缓存写入 (TTL 过期自动失效, Memory.add/update/delete 时手动清除)
        with self._cache_lock:
            self._cache[cache_key] = (time.time() + self._cache_ttl, final)

        logger.info(
            "hybrid_search_done",
            user_id=user_id,
            candidates=len(vector_results),
            vector_hits=len(vector_results),
            bm25_hits=len(keyword_rank),
            returned=len(final),
        )
        return final
