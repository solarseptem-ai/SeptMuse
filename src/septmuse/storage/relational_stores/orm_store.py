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
"""ORMMemoryStore — SQLModel ORM 记忆存储 facade。

委托 MemoryRepository 处理全部 SQL CRUD，自身只协调：
- vector_store / keyword_index 双写
- engine 暴露 (duck typing for TypedMemoryStore 共享)

一套代码跑 SQLite/MySQL/PostgreSQL。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel

from septmuse.core.logging import get_logger
from septmuse.core.utils import utcnow_iso
from septmuse.services.database.models import HistoryTable, MemoryTable
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.repositories.memory import MemoryRepository

logger = get_logger(__name__)


class ORMMemoryStore(MemoryStore):
    """SQLModel ORM 记忆存储 facade — 跨方言 CRUD。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        store = ORMMemoryStore(engine)
        mid = store.add("hello", [0.1, 0.2], user_id="alice")
    """

    def __init__(
        self,
        engine: Engine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
        owns_engine: bool = True,
    ) -> None:
        self._engine = engine
        self._session_maker = sessionmaker(engine, class_=Session, expire_on_commit=False)
        self._owns_engine = owns_engine
        self._repo = MemoryRepository(self._session_maker)
        if vector_store is None:
            from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

            vector_store = SQLAlchemyVectorStore(engine)
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        if self._owns_engine:
            self._create_tables()
        logger.info("orm_store_ready", dialect=engine.dialect.name, owns_engine=owns_engine)

    @property
    def engine(self) -> Engine:
        """暴露内部 engine，供 facade duck typing 取用。"""
        return self._engine

    def _create_tables(self) -> None:
        """建表 — SQLModel.metadata.create_all 跨方言 DDL + MigrationRunner 补旧 DB 缺列。"""
        SQLModel.metadata.create_all(self._engine)
        from septmuse.storage.migrations.runner import MigrationRunner

        MigrationRunner(self._engine).run()

    def close(self) -> None:
        """释放引擎资源 (仅当 owns_engine=True 时 dispose)。"""
        if self._owns_engine:
            self._engine.dispose()

    def reset(self) -> None:
        """重置存储 (删表数据 + 双写清理)."""
        with contextlib.suppress(Exception):
            self._repo._session_maker.bind = self._engine  # type: ignore[attr-defined]
        # 直接执行 DELETE，不走 Repository（避免 Repository 暴露 reset 语义）
        with self._repo._session_maker() as session:  # type: ignore[call-arg]
            for table_name in ["memories", "history", "memory_access_logs"]:
                with contextlib.suppress(Exception):
                    session.exec(text(f"DELETE FROM {table_name}"))  # type: ignore[attr-defined]
            session.commit()  # type: ignore[attr-defined]
        # 双写清理: vector_store
        if self._vector_store is not None:
            try:
                if hasattr(self._vector_store, "reset_collection"):
                    self._vector_store.reset_collection()
                elif hasattr(self._vector_store, "delete_collection"):
                    self._vector_store.delete_collection()
            except Exception:
                pass
        # 双写清理: keyword_index
        if self._keyword_index is not None:
            try:
                if hasattr(self._keyword_index, "reset"):
                    self._keyword_index.reset()
            except Exception:
                pass
        logger.info("orm_store_reset_done")

    # ------------------------------------------------------------------
    # 核心 CRUD — facade 协调 Repository + vector/keyword 双写
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆, 返回 memory_id。"""
        mid = f"mem-{uuid.uuid4()}"
        self._repo.add(
            mid, content,
            user_id=user_id, agent_id=agent_id, session_id=session_id,
            metadata=metadata, valid_at=valid_at,
        )
        # 双写: vector_store + keyword_index
        if self._vector_store is not None:
            self._vector_store.insert_vectors(
                [embedding], [mid],
                [{"user_id": user_id, "agent_id": agent_id, "session_id": session_id}],
            )
        if self._keyword_index is not None:
            self._keyword_index.add_docs({mid: content})
        logger.info("memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    def add_batch(
        self,
        records: list[tuple[str, list[float]]],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> list[str]:
        """批量添加记忆, 返回 memory_id 列表 (单次 commit)。"""
        if not records:
            return []
        now = utcnow_iso()
        mids: list[str | None] = [None] * len(records)
        mem_rows: list[MemoryTable] = []
        hist_rows: list[HistoryTable] = []
        all_embeddings: list[list[float]] = []
        all_ids: list[str] = []
        all_docs: dict[str, str] = {}
        seen_hashes: set[str] = set()

        for idx, (content, embedding) in enumerate(records):
            h = hashlib.md5(content.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            mid = f"mem-{uuid.uuid4()}"
            mids[idx] = mid
            mem_rows.append(MemoryTable(
                id=mid,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                metadata_json=json.dumps(metadata or {}),
                created_at=now,
                updated_at=now,
                valid_at=valid_at,
                is_deleted=0,
                state="active",
            ))
            hist_rows.append(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=mid,
                old_memory=None,
                new_memory=content,
                event="ADD",
                created_at=now,
                is_deleted=0,
            ))
            all_embeddings.append(embedding)
            all_ids.append(mid)
            all_docs[mid] = content

        if not mem_rows:
            logger.info("memory_batch_all_duplicates", count=len(records), user_id=user_id)
            return mids  # type: ignore[return-value]

        self._repo.add_batch(mids, mem_rows, hist_rows)

        # 批量双写
        if self._vector_store is not None and all_embeddings:
            self._vector_store.insert_vectors(
                all_embeddings, all_ids,
                [{"user_id": user_id, "agent_id": agent_id, "session_id": session_id}] * len(all_ids),
            )
        if self._keyword_index is not None and all_docs:
            self._keyword_index.add_docs(all_docs)
        logger.info(
            "memory_batch_added", count=len(all_ids), duplicates=len(records) - len(all_ids), user_id=user_id
        )
        return mids  # type: ignore[return-value]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        return self._repo.get(memory_id)

    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索 (委托给 _vector_store ANN 索引 + Repository 查内容)。"""
        if self._vector_store is None:
            return []

        effective_session_id = session_id
        if effective_session_id is None and filters:
            effective_session_id = filters.get("session_id")

        vs_filters: dict[str, Any] = {"user_id": user_id}
        if effective_session_id is not None:
            vs_filters["session_id"] = effective_session_id

        vec_results = self._vector_store.search_vectors(
            query_embedding, top_k=top_k * 5, filters=vs_filters
        )
        if not vec_results:
            return []

        vec_results = [r for r in vec_results if r.score >= threshold]
        if not vec_results:
            return []

        score_map = {r.id: r.score for r in vec_results}
        rows = self._repo.search_by_ids(
            list(score_map.keys()),
            session_id=effective_session_id,
            filters=filters,
        )

        results: list[dict[str, Any]] = []
        for mem in rows:
            results.append({
                "id": mem.id,
                "memory": mem.content,
                "score": score_map.get(mem.id, 0.0),
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        return self._repo.get_all(user_id=user_id, session_id=session_id, filters=filters)

    def delete(self, memory_id: str) -> None:
        """软删除 (Repository SQL + vector/keyword 清理)。"""
        old_content = self._repo.delete(memory_id)
        if old_content is None:
            return
        if self._vector_store is not None:
            self._vector_store.delete_vector(memory_id)
        if self._keyword_index is not None:
            self._keyword_index.delete_docs([memory_id])
        logger.info("memory_deleted", memory_id=memory_id)

    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        result = self._repo.update(memory_id, content, metadata=metadata)
        if result is None:
            return False
        # 双写更新
        if self._vector_store is not None:
            self._vector_store.update_vector(
                memory_id, embedding,
                payload={
                    "user_id": result["user_id"],
                    "agent_id": result["agent_id"],
                    "session_id": result["session_id"],
                },
            )
        if self._keyword_index is not None:
            self._keyword_index.add_docs({memory_id: content})
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        return self._repo.get_history(memory_id)

    def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志。"""
        return self._repo.record_access_log(memory_id, app_id, access_type, metadata)

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志。"""
        return self._repo.get_access_logs(memory_id, limit)

    # ------------------------------------------------------------------
    # 时态 — 全委托 Repository
    # ------------------------------------------------------------------

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真。"""
        existing = self._repo.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        result = self._repo.invalidate(memory_id, invalid_at=invalid_at)
        if result is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        logger.info("memory_invalidated", memory_id=memory_id, invalid_at=result["invalid_at"])
        return result

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。"""
        return self._repo.get_temporal_valid(reference_time, user_id=user_id, session_id=session_id)

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。"""
        return self._repo.get_temporal_interval(start, end, user_id=user_id, session_id=session_id)

    # ------------------------------------------------------------------
    # 关键词检索 — facade 协调 keyword_index + Repository
    # ------------------------------------------------------------------

    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。无 keyword_index 时返回空。"""
        if self._keyword_index is None:
            return []
        scores = self._keyword_index.retrieve(query, top_k * 2)
        if not scores:
            return []
        results: list[dict[str, Any]] = []
        for doc_id, score in scores.items():
            mem = self._repo.get_by_id_for_keyword(doc_id, user_id=user_id, session_id=session_id)
            if mem is not None:
                mem["score"] = float(score)
                results.append(mem)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 关系查询 (跨 agent 共享) — 全委托 Repository
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        return self._repo.list_agents(user_id)

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        return self._repo.list_users(agent_id)

    def get_user_ids(self) -> list[str]:
        """列出所有有活跃记忆的 user_id (去重)。"""
        return self._repo.get_user_ids()

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆。"""
        return self._repo.get_shared_memories(user_id, limit)
