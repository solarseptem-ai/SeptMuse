"""实体向量库 (独立 SQLite 表, 借鉴 mem0 V3 去图化设计)。

通过 SQLAlchemy engine 构造 (ORMMemoryStore 路径, SQLModel ORM)。
embedder 可选——有则做语义去重 (score >= 0.95), 无则只精确匹配。
"""

from __future__ import annotations

import contextlib
import json
import struct
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, SQLModel, select

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.extraction.entity import Entity, _normalize_entity_text
from septmuse.services.database.models.entity import EntityTable

logger = get_logger(__name__)


class EntityStore:
    """实体向量库 (独立 SQLite 表, 同库)。

    通过 SQLAlchemy engine + Session 构造 (ORMMemoryStore 路径)。
    embedder 可选——有则做语义去重 (score>=0.95), 无则只精确匹配。
    """

    @classmethod
    def from_engine(cls, engine, embedder: Embedder | None = None) -> EntityStore:
        """从 SQLAlchemy Engine 构造 (ORMMemoryStore 路径, SQLModel ORM)。

        用 Session(engine) + EntityTable ORM 操作。
        """
        store = cls.__new__(cls)
        store._engine = engine
        store._embedder = embedder
        SQLModel.metadata.create_all(engine)
        return store

    def upsert(
        self,
        entity: Entity,
        memory_id: str,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> str:
        """upsert 实体 (借鉴 mem0 _upsert_entity)。

        1. 精确归一化名匹配 → 命中则 linked_memory_ids 追加 memory_id
        2. 语义匹配 (embedder 有时) → score>=0.95 命中则追加
        3. 新建 → 插入实体 + 嵌入向量 + linked_memory_ids=[memory_id]

        Returns: entity_id
        """
        normalized = _normalize_entity_text(entity.text)

        # 1. 精确归一化名匹配
        existing = self._find_by_text(normalized, user_id=user_id)
        if existing:
            self._append_memory_id(existing["id"], memory_id)
            return existing["id"]

        # 2. 语义匹配 (embedder 有时)
        emb = None
        if self._embedder is not None:
            emb = self._embedder.embed(entity.text)
            semantic_match = self._find_by_embedding(emb, user_id=user_id, threshold=0.95)
            if semantic_match:
                self._append_memory_id(semantic_match["id"], memory_id)
                return semantic_match["id"]

        # 3. 新建
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        emb_blob = self._serialize_embedding(emb) if emb is not None else None
        with Session(self._engine) as session:
            row = EntityTable(
                id=entity_id,
                entity_text=entity.text,
                entity_type=entity.entity_type,
                entity_embedding=emb_blob,
                linked_memory_ids=json.dumps([memory_id]),
                user_id=user_id,
                agent_id=agent_id,
                created_at=now,
                updated_at=now,
                is_deleted=0,
            )
            session.add(row)
            session.commit()
        return entity_id

    def upsert_batch(
        self,
        items: list[tuple[Any, set[str]]],
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[str]:
        """批量 upsert (对齐 mem0 V3 Phase 7)。

        批量 embed + 批量精确匹配 + 批量语义匹配 + 批量 INSERT，
        消除逐条 embed / DB 查询 / DB 写入的 N 次往返。

        Args:
            items: [(Entity, set[memory_id]), ...] 列表
        Returns: [entity_id, ...] 与 items 同序
        """
        if not items:
            return []

        # 1. 批量 embed
        texts = [e.text for e, _ in items]
        embeddings = (
            self._embedder.embed_batch(texts) if self._embedder is not None else [None] * len(texts)
        )

        # 2. 批量精确匹配
        normalized_texts = [_normalize_entity_text(t) for t in texts]
        exact_matches = self._find_by_text_batch(normalized_texts, user_id=user_id)

        # 3. 批量语义匹配 (未精确命中的)
        semantic_matches: dict[int, dict[str, Any]] = {}
        unmatched_indices = [
            i for i in range(len(items)) if normalized_texts[i] not in exact_matches
        ]
        if unmatched_indices and self._embedder is not None:
            unmatched_embs = [embeddings[i] for i in unmatched_indices]
            semantic_matches = self._find_by_embedding_batch(
                unmatched_embs, user_id=user_id, threshold=0.95
            )
            # key 从 unmatched 局部 index 映射回全局 index
            semantic_matches = {
                unmatched_indices[k]: v for k, v in semantic_matches.items()
            }

        # 4. 分流: 命中 → append; 未命中 → insert
        results: list[str | None] = [None] * len(items)
        to_insert: list[tuple[int, Any, set[str], list[float] | None]] = []

        for i, (entity, memory_ids) in enumerate(items):
            exact = exact_matches.get(normalized_texts[i])
            semantic = semantic_matches.get(i)
            match = exact or semantic
            if match:
                self._append_memory_ids(match["id"], memory_ids)
                results[i] = match["id"]
            else:
                to_insert.append((i, entity, memory_ids, embeddings[i]))

        if to_insert:
            new_ids = self._batch_insert(to_insert, user_id=user_id, agent_id=agent_id)
            for (idx, _, _, _), eid in zip(to_insert, new_ids, strict=True):
                results[idx] = eid

        return [r for r in results]  # type: ignore[list-item]

    def _find_by_text_batch(
        self, normalized_texts: list[str], *, user_id: str
    ) -> dict[str, dict[str, Any]]:
        """批量精确归一化名匹配. Returns {normalized_text: entity_dict}。

        DB 存原始 entity_text, 需客户端归一化后比较 (对齐 _find_by_text)。
        一次 DB 查询替代 N 次。
        """
        if not normalized_texts:
            return {}
        normalized_set = set(normalized_texts)
        result: dict[str, dict[str, Any]] = {}
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            rows = session.exec(stmt).all()
        for row in rows:
            norm = _normalize_entity_text(row.entity_text)
            if norm in normalized_set and norm not in result:
                result[norm] = self._row_to_dict(row)
        return result

    def _find_by_embedding_batch(
        self,
        embeddings: list[list[float] | None],
        *,
        user_id: str,
        threshold: float = 0.95,
    ) -> dict[int, dict[str, Any]]:
        """全表扫 + numpy 矩阵余弦. Returns {input_index: entity_dict}。

        一次 DB 查询 + 一次矩阵运算替代 N 次 O(N*M) 循环。
        """
        if not embeddings:
            return {}
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            rows = session.exec(stmt).all()
        if not rows:
            return {}

        import numpy as np

        db_embs: list[list[float]] = []
        valid_rows: list[Any] = []
        for r in rows:
            emb = self._deserialize_embedding(r.entity_embedding)
            if emb is not None:
                db_embs.append(emb)
                valid_rows.append(r)

        if not valid_rows:
            return {}

        query_indices = [i for i, e in enumerate(embeddings) if e is not None]
        if not query_indices:
            return {}

        db_matrix = np.array(db_embs, dtype=np.float32)
        query_matrix = np.array(
            [embeddings[i] for i in query_indices], dtype=np.float32
        )

        # 归一化
        db_norm = db_matrix / (np.linalg.norm(db_matrix, axis=1, keepdims=True) + 1e-8)
        query_norm = query_matrix / (
            np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-8
        )
        sims = query_norm @ db_norm.T  # [n_query, n_db]

        matches: dict[int, dict[str, Any]] = {}
        for local_i, global_i in enumerate(query_indices):
            best = int(np.argmax(sims[local_i]))
            if sims[local_i][best] >= threshold:
                matches[global_i] = self._row_to_dict(valid_rows[best])
        return matches

    def _batch_insert(
        self,
        items: list[tuple[int, Any, set[str], list[float] | None]],
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[str]:
        """批量插入新实体. items: [(global_idx, Entity, memory_ids, embedding), ...]"""
        now = datetime.now(timezone.utc).isoformat()
        new_ids: list[str] = []
        with Session(self._engine) as session:
            for _, entity, memory_ids, emb in items:
                eid = str(uuid.uuid4())
                emb_blob = self._serialize_embedding(emb) if emb is not None else None
                row = EntityTable(
                    id=eid,
                    entity_text=entity.text,
                    entity_type=entity.entity_type,
                    entity_embedding=emb_blob,
                    linked_memory_ids=json.dumps(sorted(memory_ids)),
                    user_id=user_id,
                    agent_id=agent_id,
                    created_at=now,
                    updated_at=now,
                    is_deleted=0,
                )
                session.add(row)
                new_ids.append(eid)
            session.commit()
        return new_ids

    def _append_memory_ids(self, entity_id: str, memory_ids: set[str]) -> None:
        """批量追加 memory_ids (set → sorted list, 幂等)。"""
        with Session(self._engine) as session:
            row = session.get(EntityTable, entity_id)
            if not row:
                return
            existing = set(json.loads(row.linked_memory_ids)) if row.linked_memory_ids else set()
            existing.update(memory_ids)
            row.linked_memory_ids = json.dumps(sorted(existing))
            row.updated_at = datetime.now(timezone.utc).isoformat()
            session.add(row)
            session.commit()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """EntityTable row → dict。"""
        return {
            "id": row.id,
            "entity_text": row.entity_text,
            "entity_type": row.entity_type,
            "linked_memory_ids": json.loads(row.linked_memory_ids)
            if row.linked_memory_ids
            else [],
            "user_id": row.user_id,
            "agent_id": row.agent_id,
        }

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """取单条实体。"""
        with Session(self._engine) as session:
            row = session.get(EntityTable, entity_id)
            if not row or row.is_deleted:
                return None
            return {
                "id": row.id,
                "entity_text": row.entity_text,
                "entity_type": row.entity_type,
                "entity_embedding": row.entity_embedding,
                "linked_memory_ids": row.linked_memory_ids,
                "user_id": row.user_id,
                "agent_id": row.agent_id,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }

    def _find_by_text(self, normalized_text: str, *, user_id: str) -> dict[str, Any] | None:
        """精确归一化名匹配。"""
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            rows = session.exec(stmt).all()
            for row in rows:
                if _normalize_entity_text(row.entity_text) == normalized_text:
                    return {
                        "id": row.id,
                        "entity_text": row.entity_text,
                        "entity_type": row.entity_type,
                        "linked_memory_ids": row.linked_memory_ids,
                    }
        return None

    def _find_by_embedding(
        self, embedding: list[float], *, user_id: str, threshold: float = 0.95
    ) -> dict[str, Any] | None:
        """语义匹配 (cosine similarity >= threshold)。"""
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            rows = session.exec(stmt).all()
            for row in rows:
                if row.entity_embedding is None:
                    continue
                stored_emb = self._deserialize_embedding(row.entity_embedding)
                if stored_emb is not None:
                    sim = _cosine_similarity(embedding, stored_emb)
                    if sim >= threshold:
                        return {
                            "id": row.id,
                            "entity_text": row.entity_text,
                            "entity_type": row.entity_type,
                            "linked_memory_ids": row.linked_memory_ids,
                        }
        return None

    def _append_memory_id(self, entity_id: str, memory_id: str) -> None:
        """向实体的 linked_memory_ids 追加 memory_id。"""
        with Session(self._engine) as session:
            row = session.get(EntityTable, entity_id)
            if not row:
                return
            linked = json.loads(row.linked_memory_ids)
            if memory_id not in linked:
                linked.append(memory_id)
                row.linked_memory_ids = json.dumps(linked)
            row.updated_at = datetime.now(timezone.utc).isoformat()
            session.add(row)
            session.commit()

    @staticmethod
    def _serialize_embedding(emb: list[float]) -> bytes:
        """序列化嵌入向量为 BLOB。"""
        return struct.pack(f"{len(emb)}f", *emb)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> list[float] | None:
        """反序列化嵌入向量。"""
        if not blob:
            return None
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体: 精确匹配 + 向量相似度 (embedder 有时)。

        Returns: [{"id","entity_text","entity_type","linked_memory_ids","score"}]
        """
        results: list[dict[str, Any]] = []
        normalized_query = _normalize_entity_text(query)

        # 精确匹配
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            rows = session.exec(stmt).all()
            for row in rows:
                normalized_entity = _normalize_entity_text(row.entity_text)
                if normalized_query in normalized_entity or normalized_entity in normalized_query:
                    results.append(
                        {
                            "id": row.id,
                            "entity_text": row.entity_text,
                            "entity_type": row.entity_type,
                            "linked_memory_ids": row.linked_memory_ids,
                            "score": 1.0 if normalized_entity == normalized_query else 0.8,
                        }
                    )

        # 向量相似度 (embedder 有时)
        if self._embedder is not None:
            query_emb = self._embedder.embed(query)
            with Session(self._engine) as session:
                stmt = select(EntityTable).where(
                    EntityTable.user_id == user_id,
                    EntityTable.is_deleted == 0,
                )
                rows = session.exec(stmt).all()
                existing_ids = {r["id"] for r in results}
                for row in rows:
                    if row.id in existing_ids:
                        continue
                    if row.entity_embedding is None:
                        continue
                    stored_emb = self._deserialize_embedding(row.entity_embedding)
                    if stored_emb is not None:
                        sim = _cosine_similarity(query_emb, stored_emb)
                        if sim > 0.3:
                            results.append(
                                {
                                    "id": row.id,
                                    "entity_text": row.entity_text,
                                    "entity_type": row.entity_type,
                                    "linked_memory_ids": row.linked_memory_ids,
                                    "score": sim,
                                }
                            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部未删除实体。"""
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,
            )
            if entity_type:
                stmt = stmt.where(EntityTable.entity_type == entity_type)
            stmt = stmt.order_by(EntityTable.created_at.desc()).limit(limit)
            rows = session.exec(stmt).all()
        return [
            {
                "id": row.id,
                "entity_text": row.entity_text,
                "entity_type": row.entity_type,
                "linked_memory_ids": row.linked_memory_ids,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    def get_linked_memories(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        with Session(self._engine) as session:
            row = session.get(EntityTable, entity_id)
            if not row or row.is_deleted:
                return []
            return json.loads(row.linked_memory_ids)

    def remove_memory_from_entities(self, memory_id: str) -> None:
        """删除记忆时清理实体引用 (借鉴 mem0 _remove_memory_from_entity_store)。

        memory_id 是 UUID 全局唯一, 不需 user_id 过滤。
        1. 查 linked_memory_ids 包含 memory_id 的实体
        2. 移除 memory_id
        3. linked_memory_ids 空 → 软删除实体
        """
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(EntityTable.is_deleted == 0)
            rows = session.exec(stmt).all()
            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                linked = json.loads(row.linked_memory_ids)
                if memory_id not in linked:
                    continue
                remaining = [mid for mid in linked if mid != memory_id]
                if not remaining:
                    row.is_deleted = 1
                    row.updated_at = now
                else:
                    row.linked_memory_ids = json.dumps(remaining)
                    row.updated_at = now
                session.add(row)
            session.commit()

    def reset(self) -> None:
        """重置实体存储 (清表: septmuse_entities + entity_relations)."""
        from sqlalchemy import text

        with Session(self._engine) as session:
            for table_name in ["septmuse_entities", "entity_relations"]:
                with contextlib.suppress(Exception):
                    session.exec(text(f"DELETE FROM {table_name}"))
            session.commit()

    def close(self) -> None:
        """释放资源 (同库, 实际不关 conn)。"""
        pass


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """cosine 相似度 (score 统一为相似度 [0,1])。"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
