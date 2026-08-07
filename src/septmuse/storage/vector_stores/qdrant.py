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
"""Qdrant 向量存储 (extras, lazy import)。

支持三种连接模式: 本地嵌入 (path) / 远程 (host:port) / URL+api_key。
BM25 稀疏向量检索依赖 fastembed (可选), 未安装时自动降级为纯稠密检索。
upsert/query_points/retrieve/scroll + Filter/FieldCondition 模式。
Qdrant Distance.COSINE 返回的 score 直接是 [0,1] 相似度, 无需转换。
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any, ClassVar

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)

# payload 内部字段: 存储原始字符串 ID (Qdrant 仅接受 uint/UUID, 非UUID字符串需转换)
_ORIG_ID_KEY = "_septmuse_id"


class QdrantVectorStore(VectorStoreBase):
    """Qdrant 向量存储 (本地嵌入 / 远程服务, 需 qdrant-client)。

    三种连接模式:
        - 本地嵌入: QdrantVectorStore(path="~/.septmuse/qdrant")
        - 远程: QdrantVectorStore(host="localhost", port=6333)
        - URL: QdrantVectorStore(url="https://cluster.qdrant.io", api_key="...")

    BM25 关键词检索 (可选):
        - enable_bm25=True (默认) 时, collection 创建 bm25 稀疏向量槽位
        - 需 fastembed 库做查询编码; 未安装时 keyword_search 返回 None
        - 远程模式下自动创建 user_id/agent_id/session_id payload 索引

    用法:
        store = QdrantVectorStore(host="localhost", port=6333)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})

    Note: qdrant_client 必须 lazy import (extras 未安装时不能破坏 ``import septmuse``)。
    Collection 在首次 insert/search 时按向量维度自动创建 (Distance.COSINE)。

    本地模式客户端缓存: Qdrant 本地存储使用 portalocker 文件锁, 同一 path 仅允许
    一个 QdrantClient 实例。同进程内多个 QdrantVectorStore (如 create_app 创建
    sync + async Memory) 复用同一 client, 避免 AlreadyLocked 冲突。
    """

    # 本地模式客户端缓存: {path: QdrantClient}, 同 path 复用避免文件锁冲突
    _local_client_cache: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        collection_name: str = "septmuse",
        embedding_model_dims: int = 512,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        enable_bm25: bool = True,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self._dim = embedding_model_dims
        self._enable_bm25 = enable_bm25
        self._has_bm25_slot = False
        self._bm25_encoder: Any = None  # None=未加载, False=不可用 sentinel
        self._collection_ensured = False

        if host or url:
            # 远程模式
            self.is_local = False
            if url:
                kwargs: dict[str, Any] = {"url": url}
            else:
                kwargs = {"host": host}
                if port is not None:
                    kwargs["port"] = port
            if api_key:
                kwargs["api_key"] = api_key
            self.client = QdrantClient(**kwargs)
        else:
            # 本地嵌入模式: 同 path 复用客户端 (Qdrant 本地文件锁限制单进程单客户端)
            self.is_local = True
            local_path = path or "~/.septmuse/qdrant"
            cached = QdrantVectorStore._local_client_cache.get(local_path)
            if cached is not None:
                self.client = cached
            else:
                self.client = QdrantClient(path=local_path)
                QdrantVectorStore._local_client_cache[local_path] = self.client

        logger.info(
            "qdrant_vector_store_ready",
            collection=collection_name,
            mode="local" if self.is_local else "remote",
            bm25=enable_bm25,
        )

    @classmethod
    def clear_client_cache(cls) -> None:
        """清空本地模式客户端缓存 (测试隔离用: 关闭并移除所有缓存的 QdrantClient)。"""
        for client in cls._local_client_cache.values():
            with contextlib.suppress(Exception):
                client.close()
        cls._local_client_cache.clear()

    # ── Collection 管理 ──────────────────────────────────────────

    def _ensure_collection(self, dim: int) -> None:
        """确保 collection 存在 (按 dim 自动选择/创建, 含可选 BM25 稀疏槽位)。

        策略: 不同维度的 embedding 模型使用独立的 Qdrant collection。
        - 记录上次确认的维度, 维度变化时重新检测 (keyword_search 用 512 调用后
          search_vectors 用 1536 调用时能正确切换)
        - 优先查原 collection_name (向后兼容, 无后缀)
        - 若原 collection 存在但维度不匹配, 自动切换到 {name}_{dim} 后缀 collection
        - 维度匹配时直接使用原 collection
        """
        if self._collection_ensured and dim == self._dim:
            return
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        try:
            collections = self.client.get_collections().collections
            base_exists = any(c.name == self.collection_name for c in collections)

            if not base_exists:
                self._create_collection(dim, SparseVectorParams, VectorParams, Distance)
                self._create_payload_indexes()
                self._dim = dim
                self._collection_ensured = True
                return

            info = self.client.get_collection(self.collection_name)
            existing_dim = getattr(info.config.params.vectors, "size", None)
            if existing_dim is None or existing_dim == dim:
                sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
                self._has_bm25_slot = bool(sparse_vectors) and "bm25" in (sparse_vectors or {})
                self._create_payload_indexes()
                self._dim = dim
                self._collection_ensured = True
                return

            # 维度不匹配: 切换到 {name}_{dim} 后缀 collection
            suffixed = f"{self.collection_name}_{dim}"
            suffixed_exists = any(c.name == suffixed for c in collections)
            if suffixed_exists:
                logger.info(
                    "qdrant_dim_switch_suffixed",
                    original=self.collection_name,
                    using=suffixed,
                    existing_dim=existing_dim,
                    expected_dim=dim,
                )
                self.collection_name = suffixed
                self._dim = dim
                self._create_payload_indexes()
                self._collection_ensured = True
                return

            logger.warning(
                "qdrant_dim_mismatch_new_collection",
                original=self.collection_name,
                new=suffixed,
                existing_dim=existing_dim,
                expected_dim=dim,
                hint="切换了 embedding 模型, 使用新的 collection 名称, 旧向量数据保留在原 collection 中",
            )
            self.collection_name = suffixed
            self._create_collection(dim, SparseVectorParams, VectorParams, Distance)
            self._create_payload_indexes()
            self._dim = dim
            self._collection_ensured = True
        except Exception:
            self._collection_ensured = False
            raise

    def _create_collection(self, dim: int, SparseVectorParams: Any, VectorParams: Any, Distance: Any) -> None:
        from qdrant_client.models import Modifier

        sparse_config: dict[str, Any] | None = None
        if self._enable_bm25:
            sparse_config = {"bm25": SparseVectorParams(modifier=Modifier.IDF)}
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            sparse_vectors_config=sparse_config,
        )
        self._has_bm25_slot = self._enable_bm25

    def _create_payload_indexes(self) -> None:
        """为常用过滤字段创建 payload 索引 (仅远程模式)。"""
        if self.is_local:
            return
        from qdrant_client.models import PayloadSchemaType

        for field in ("user_id", "agent_id", "session_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.debug("qdrant_payload_index_skipped", field=field, error=str(e))

    # ── Filter 构建 (10 操作符 + AND/OR/NOT 递归) ────────────────

    def _build_filter(self, filters: dict[str, Any] | None) -> Any:
        """解析过滤字典 → Qdrant Filter 对象。空/None 返回 None。

        支持 10 个操作符: eq/ne/gt/gte/lt/lte/in/nin/contains/icontains
        支持逻辑组合: AND(必须全中) / OR(任一中) / NOT(都不中)
        支持递归嵌套: {"AND": [{"OR": [{"a": 1}, {"b": 2}]}, {"c": 3}]}
        """
        if not filters or not isinstance(filters, dict):
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must: list[Any] = []
        should: list[Any] = []
        must_not: list[Any] = []

        for key, value in filters.items():
            if key == "AND":
                for sub in value:
                    built = self._build_filter(sub)
                    if built is not None:
                        must.append(built)
            elif key == "OR":
                for sub in value:
                    built = self._build_filter(sub)
                    if built is not None:
                        should.append(built)
            elif key == "NOT":
                for sub in value:
                    built = self._build_filter(sub)
                    if built is not None:
                        must_not.append(built)
            elif isinstance(value, dict):
                cond = self._build_condition(key, value)
                if cond is not None:
                    must.append(cond)
            else:
                # 简单精确匹配: {"k": "v"}
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if not must and not should and not must_not:
            return None
        kwargs: dict[str, Any] = {}
        if must:
            kwargs["must"] = must
        if should:
            kwargs["should"] = should
        if must_not:
            kwargs["must_not"] = must_not
        return Filter(**kwargs)

    @staticmethod
    def _build_condition(key: str, ops: dict[str, Any]) -> Any:
        """从单个字段的操作符字典构建 FieldCondition。

        {"eq": v} → MatchValue    {"ne": v} → MatchExcept
        {"gt": N} → Range(gt=N)   {"gte": N} → Range(gte=N)
        {"lt": N} → Range(lt=N)   {"lte": N} → Range(lte=N)
        {"in": [...]} → MatchAny  {"nin": [...]} → MatchExcept
        {"contains": "x"} → MatchText  {"icontains": "x"} → MatchText
        """
        from qdrant_client.models import FieldCondition, MatchAny, MatchExcept, MatchText, MatchValue, Range

        if "eq" in ops:
            return FieldCondition(key=key, match=MatchValue(value=ops["eq"]))
        if "ne" in ops:
            return FieldCondition(key=key, match=MatchExcept(**{"except": [ops["ne"]]}))
        if "in" in ops:
            return FieldCondition(key=key, match=MatchAny(any=ops["in"]))
        if "nin" in ops:
            return FieldCondition(key=key, match=MatchExcept(**{"except": ops["nin"]}))
        if "contains" in ops:
            return FieldCondition(key=key, match=MatchText(text=str(ops["contains"])))
        if "icontains" in ops:
            return FieldCondition(key=key, match=MatchText(text=str(ops["icontains"])))

        # 范围操作符
        range_kwargs: dict[str, float] = {}
        for op in ("gt", "gte", "lt", "lte"):
            if op in ops:
                range_kwargs[op] = ops[op]
        if range_kwargs:
            return FieldCondition(key=key, range=Range(**range_kwargs))

        logger.warning("qdrant_unknown_filter_operator", key=key, ops=list(ops.keys()))
        return None

    # ── BM25 编码 (fastembed 可选) ───────────────────────────────

    def _get_bm25_encoder(self) -> Any:
        """延迟加载 fastembed BM25 编码器。未装返回 None。"""
        if self._bm25_encoder is None:
            try:
                from fastembed import SparseTextEmbedding

                self._bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
            except ImportError:
                self._bm25_encoder = False  # sentinel: 不可用
        return self._bm25_encoder if self._bm25_encoder is not False else None

    def _encode_bm25(self, text: str) -> Any:
        """单条文本 → BM25 SparseVector。fastembed 不可用返回 None。"""
        encoder = self._get_bm25_encoder()
        if encoder is None:
            return None
        from qdrant_client.models import SparseVector

        results = list(encoder.embed([text]))
        if results:
            s = results[0]
            return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        return None

    def _batch_encode_bm25(self, payloads: list[dict[str, Any]]) -> list[Any]:
        """批量 BM25 编码。fastembed 不可用返回全 None 列表。"""
        encoder = self._get_bm25_encoder()
        if encoder is None:
            return [None] * len(payloads)
        from qdrant_client.models import SparseVector

        texts = [p.get("data", "") or p.get("text", "") or str(p) for p in payloads]
        results = list(encoder.embed(texts))
        return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) if r else None for r in results]

    # ── 向量工具 ─────────────────────────────────────────────────

    @staticmethod
    def _extract_vector(raw: Any) -> list[float]:
        """从 Qdrant 返回的 vector 字段提取稠密向量列表。"""
        if raw is None:
            return []
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if isinstance(raw, dict):
            for key in ("", "dense", "vector"):
                value = raw.get(key)
                if value is not None:
                    return [float(x) for x in value]
        return []

    # ── ID 转换 (Qdrant 仅接受 uint/UUID, SeptMuse 用任意字符串 ID) ──

    @staticmethod
    def _to_qdrant_id(id_value: Any) -> Any:
        """转换字符串/整数 ID 为 Qdrant 可接受的 ID (uint 或 UUID 字符串)。

        - 整数 → 原样返回 (Qdrant 接受 uint64)
        - 合法 UUID 字符串 → 原样返回
        - 非法 UUID 字符串 (如 "m1") → uuid5 哈希 (确定性, 可重复)
        """
        if isinstance(id_value, int):
            return id_value
        s = str(id_value)
        try:
            uuid.UUID(s)
            return s
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

    @staticmethod
    def _inject_orig_id(payload: dict[str, Any] | None, orig_id: Any) -> dict[str, Any]:
        """创建 payload 副本并注入原始 ID (不修改原 dict)。"""
        p = dict(payload or {})
        p[_ORIG_ID_KEY] = str(orig_id)
        return p

    @staticmethod
    def _extract_orig_id(point_id: Any, payload: dict[str, Any] | None) -> str:
        """从 payload 提取原始 ID, 没有则回退到 Qdrant point ID。"""
        if payload and _ORIG_ID_KEY in payload:
            return str(payload[_ORIG_ID_KEY])
        return str(point_id)

    @staticmethod
    def _clean_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """移除内部 _septmuse_id 字段, 返回用户可见的 payload。"""
        if not payload:
            return payload
        p = dict(payload)
        p.pop(_ORIG_ID_KEY, None)
        return p

    def _to_search_result(self, point: Any) -> VectorSearchResult:
        """Qdrant ScoredPoint → VectorSearchResult (提取原始 ID, 清理 payload)。"""
        return VectorSearchResult(
            id=self._extract_orig_id(point.id, point.payload),
            score=float(point.score),
            payload=self._clean_payload(point.payload),
        )

    def _to_vector_entry(self, point: Any) -> VectorEntry:
        """Qdrant Record → VectorEntry (提取原始 ID, 清理 payload)。"""
        return VectorEntry(
            id=self._extract_orig_id(point.id, point.payload),
            vector=self._extract_vector(point.vector),
            payload=self._clean_payload(point.payload),
        )

    # ── CRUD (VectorStoreBase 抽象方法实现) ──────────────────────

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) length mismatch")
        if payloads is None:
            payloads = [{}] * len(ids)
        elif len(payloads) != len(ids):
            raise ValueError(f"payloads ({len(payloads)}) and ids ({len(ids)}) length mismatch")
        if not vectors:
            return
        self._ensure_collection(len(vectors[0]))
        from qdrant_client.models import PointStruct

        # 批量预计算 BM25 稀疏向量
        sparse_vecs = self._batch_encode_bm25(payloads) if self._has_bm25_slot else [None] * len(ids)

        points = []
        for vid, vec, payload, sparse in zip(ids, vectors, payloads, sparse_vecs, strict=True):
            qdrant_id = self._to_qdrant_id(vid)
            if sparse is not None:
                point_vec: Any = {"": vec, "bm25": sparse}
            else:
                point_vec = vec
            points.append(
                PointStruct(id=qdrant_id, vector=point_vec, payload=self._inject_orig_id(payload, vid))
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_collection(len(query_vector))
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_search_result(p) for p in hits.points]

    def search_batch(
        self,
        queries: list[str],
        vectors_list: list[list[float]],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[VectorSearchResult]]:
        """批量搜索 (Qdrant 原生 query_batch_points, 出错降级为顺序搜索)。"""
        if not vectors_list:
            return []
        self._ensure_collection(len(vectors_list[0]))
        query_filter = self._build_filter(filters)
        from qdrant_client.models import QueryRequest

        requests = [
            QueryRequest(query=vec, filter=query_filter, limit=top_k, with_payload=True)
            for vec in vectors_list
        ]
        try:
            results = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )
            return [
                [self._to_search_result(p) for p in batch.points]
                for batch in results
            ]
        except Exception as e:
            logger.warning("qdrant_batch_fallback_sequential", error=str(e))
            return [self.search_vectors(v, top_k, filters) for v in vectors_list]

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult] | None:
        """BM25 稀疏向量检索。无 bm25 槽位或 fastembed 不可用 → 返回 None。"""
        self._ensure_collection(self._dim)
        if not self._has_bm25_slot:
            return None
        sparse = self._encode_bm25(query)
        if sparse is None:
            return None
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse,
            using="bm25",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_search_result(p) for p in hits.points]

    def update_vector(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """原地更新单条向量 + payload。不存在返回 False。"""
        self._ensure_collection(len(vector))
        qdrant_id = self._to_qdrant_id(vector_id)
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[qdrant_id],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        from qdrant_client.models import PointStruct

        if self._has_bm25_slot:
            text = (payload or {}).get("data", "") or (payload or {}).get("text", "") or str(payload)
            sparse = self._encode_bm25(text)
            if sparse is not None:
                point_vec: Any = {"": vector, "bm25": sparse}
            else:
                point_vec = vector
        else:
            point_vec = vector
        point = PointStruct(
            id=qdrant_id,
            vector=point_vec,
            payload=self._inject_orig_id(payload, vector_id),
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return True

    def delete_vector(self, vector_id: str) -> bool:
        qdrant_id = self._to_qdrant_id(vector_id)
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[qdrant_id],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[qdrant_id]),
        )
        return True

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        qdrant_id = self._to_qdrant_id(vector_id)
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[qdrant_id],
            with_payload=True,
            with_vectors=True,
        )
        if not result:
            return None
        return self._to_vector_entry(result[0])

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        query_filter = self._build_filter(filters)
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit if limit is not None else 100,
            with_payload=True,
            with_vectors=True,
        )
        points, _ = result
        return [self._to_vector_entry(point) for point in points]

    # ── Collection 管理 (VectorStoreBase 覆写) ───────────────────

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def delete_collection(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)
        self._collection_ensured = False
        self._has_bm25_slot = False
        logger.info("qdrant_collection_deleted", collection=self.collection_name)

    def get_collection_info(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            return {"name": self.collection_name, "count": 0, "dim": 0, "distance": "COSINE"}
        count = info.points_count or 0
        vectors = info.config.params.vectors
        dim = 0
        distance = "COSINE"
        if hasattr(vectors, "size"):
            dim = vectors.size
            distance = vectors.distance.name if hasattr(vectors, "distance") else "COSINE"
        return {"name": self.collection_name, "count": count, "dim": dim, "distance": distance}

    def reset_collection(self) -> None:
        self.delete_collection()

    def close(self) -> None:
        self.client.close()
        logger.info("qdrant_vector_store_closed", collection=self.collection_name)
