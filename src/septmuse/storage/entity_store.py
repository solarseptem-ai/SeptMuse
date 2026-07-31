"""实体向量库 (独立 SQLite 表, 借鉴 mem0 V3 去图化设计)。

复用 SQLiteMemoryStore 的 conn + lock (类似 SQLiteGraphStore 模式)。
embedder 可选——有则做语义去重 (score >= 0.95), 无则只精确匹配。
"""

from __future__ import annotations

import json
import struct
import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.extraction.entity import Entity, _normalize_entity_text

logger = get_logger(__name__)


class EntityStore:
    """实体向量库 (独立 SQLite 表, 同库)。

    复用 SQLiteMemoryStore 的 conn + lock。
    embedder 可选——有则做语义去重 (score>=0.95), 无则只精确匹配。
    """

    def __init__(self, conn, lock, embedder: Embedder | None = None):
        self._conn = conn
        self._lock = lock
        self._embedder = embedder
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS septmuse_entities (
                    id TEXT PRIMARY KEY,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_embedding BLOB,
                    linked_memory_ids TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_deleted INTEGER DEFAULT 0,
                    UNIQUE(user_id, entity_text)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_user ON septmuse_entities(user_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_text ON septmuse_entities(entity_text)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_deleted ON septmuse_entities(is_deleted)")
            self._conn.commit()

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
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO septmuse_entities
                    (id, entity_text, entity_type, entity_embedding, linked_memory_ids,
                     user_id, agent_id, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    entity_id,
                    entity.text,
                    entity.entity_type,
                    emb_blob,
                    json.dumps([memory_id]),
                    user_id,
                    agent_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return entity_id

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """取单条实体。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids, "
                "user_id, agent_id, created_at, updated_at FROM septmuse_entities "
                "WHERE id=? AND is_deleted=0",
                (entity_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "entity_text": r[1],
            "entity_type": r[2],
            "entity_embedding": r[3],
            "linked_memory_ids": r[4],
            "user_id": r[5],
            "agent_id": r[6],
            "created_at": r[7],
            "updated_at": r[8],
        }

    def _find_by_text(self, normalized_text: str, *, user_id: str) -> dict[str, Any] | None:
        """精确归一化名匹配。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, linked_memory_ids FROM septmuse_entities "
                "WHERE user_id=? AND is_deleted=0",
                (user_id,),
            )
            for r in cur.fetchall():
                if _normalize_entity_text(r[1]) == normalized_text:
                    return {
                        "id": r[0],
                        "entity_text": r[1],
                        "entity_type": r[2],
                        "linked_memory_ids": r[3],
                    }
        return None

    def _find_by_embedding(
        self, embedding: list[float], *, user_id: str, threshold: float = 0.95
    ) -> dict[str, Any] | None:
        """语义匹配 (cosine similarity >= threshold)。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids "
                "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_embedding IS NOT NULL",
                (user_id,),
            )
            for r in cur.fetchall():
                stored_emb = self._deserialize_embedding(r[3])
                if stored_emb is not None:
                    sim = _cosine_similarity(embedding, stored_emb)
                    if sim >= threshold:
                        return {
                            "id": r[0],
                            "entity_text": r[1],
                            "entity_type": r[2],
                            "linked_memory_ids": r[4],
                        }
        return None

    def _append_memory_id(self, entity_id: str, memory_id: str) -> None:
        """向实体的 linked_memory_ids 追加 memory_id。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT linked_memory_ids FROM septmuse_entities WHERE id=?",
                (entity_id,),
            )
            r = cur.fetchone()
            if not r:
                return
            linked = json.loads(r[0])
            if memory_id not in linked:
                linked.append(memory_id)
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE septmuse_entities SET linked_memory_ids=?, updated_at=? WHERE id=?",
                (json.dumps(linked), now, entity_id),
            )
            self._conn.commit()

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
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, linked_memory_ids FROM septmuse_entities "
                "WHERE user_id=? AND is_deleted=0",
                (user_id,),
            )
            for r in cur.fetchall():
                normalized_entity = _normalize_entity_text(r[1])
                if normalized_query in normalized_entity or normalized_entity in normalized_query:
                    results.append(
                        {
                            "id": r[0],
                            "entity_text": r[1],
                            "entity_type": r[2],
                            "linked_memory_ids": r[3],
                            "score": 1.0 if normalized_entity == normalized_query else 0.8,
                        }
                    )

        # 向量相似度 (embedder 有时)
        if self._embedder is not None:
            query_emb = self._embedder.embed(query)
            with self._lock:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_embedding IS NOT NULL",
                    (user_id,),
                )
                existing_ids = {r["id"] for r in results}
                for r in cur.fetchall():
                    if r[0] in existing_ids:
                        continue
                    stored_emb = self._deserialize_embedding(r[3])
                    if stored_emb is not None:
                        sim = _cosine_similarity(query_emb, stored_emb)
                        if sim > 0.3:
                            results.append(
                                {
                                    "id": r[0],
                                    "entity_text": r[1],
                                    "entity_type": r[2],
                                    "linked_memory_ids": r[4],
                                    "score": sim,
                                }
                            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部未删除实体。"""
        with self._lock:
            if entity_type:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, linked_memory_ids, created_at "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_type=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, entity_type, limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, linked_memory_ids, created_at "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "entity_text": r[1],
                "entity_type": r[2],
                "linked_memory_ids": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def get_linked_memories(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT linked_memory_ids FROM septmuse_entities WHERE id=? AND is_deleted=0",
                (entity_id,),
            )
            r = cur.fetchone()
        if not r:
            return []
        return json.loads(r[0])

    def remove_memory_from_entities(self, memory_id: str) -> None:
        """删除记忆时清理实体引用 (借鉴 mem0 _remove_memory_from_entity_store)。

        memory_id 是 UUID 全局唯一, 不需 user_id 过滤。
        1. 查 linked_memory_ids 包含 memory_id 的实体
        2. 移除 memory_id
        3. linked_memory_ids 空 → 软删除实体
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, linked_memory_ids FROM septmuse_entities WHERE is_deleted=0",
            )
            for r in cur.fetchall():
                entity_id, linked_json = r[0], r[1]
                linked = json.loads(linked_json)
                if memory_id not in linked:
                    continue
                remaining = [mid for mid in linked if mid != memory_id]
                now = datetime.now(timezone.utc).isoformat()
                if not remaining:
                    self._conn.execute(
                        "UPDATE septmuse_entities SET is_deleted=1, updated_at=? WHERE id=?",
                        (now, entity_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE septmuse_entities SET linked_memory_ids=?, updated_at=? WHERE id=?",
                        (json.dumps(remaining), now, entity_id),
                    )
            self._conn.commit()

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
