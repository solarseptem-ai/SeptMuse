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
"""V2 语义记忆子组件 — SemanticFact CRUD + 向量检索 + 置信度加权。"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.memory.base import LongTermMemory
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class SemanticMemory(LongTermMemory):
    """V2 语义记忆 — 三元组 CRUD + 向量检索。"""

    def __init__(self, store: TypedMemoryStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        user_id: str,
        context: str | None = None,
        confidence: float = 1.0,
        provenance: str = "user",
        tags: list[str] | None = None,
        embed: bool = True,
    ) -> Any:
        """添加语义事实。embed=True 时自动生成向量。"""
        embedding = None
        if embed:
            text = f"{subject} {predicate} {object}"
            if context:
                text += f" {context}"
            embedding = self.embedder.embed(text)
        return self.store.add_fact(
            subject,
            predicate,
            object,
            user_id=user_id,
            context=context,
            confidence=confidence,
            provenance=provenance,
            tags=tags,
            embedding=embedding,
        )

    def search_facts(self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1) -> list[dict[str, Any]]:
        """语义检索事实 (向量 + 置信度加权)。"""
        emb = self.embedder.embed(query)
        results = self.store.search_facts(emb, user_id=user_id, top_k=top_k, threshold=threshold)
        for r in results:
            r["final_score"] = r["score"] * r.get("confidence", 1.0)
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def get_all_facts(self, *, user_id: str) -> list[Any]:
        """列出用户全部事实。"""
        return self.store.get_all_facts(user_id=user_id)

    def add_identity(self, key: str, value: str, *, user_id: str) -> Any:
        """添加身份记忆子类 (tags=["identity"])。"""
        return self.add_fact(
            subject=user_id,
            predicate="identity",
            object=f"{key}: {value}",
            user_id=user_id,
            tags=["identity"],
            confidence=1.0,
            provenance="user",
        )

    # ── LongTermMemory ABC 实现 ──

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> bool:
        """标记事实不再为真 (软删除 is_deleted=True)。"""
        return self.store.soft_delete_fact(memory_id)

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (审计用, 暂返回基本信息)。"""
        return [{"id": memory_id, "event": "no_history_available"}]

    def get_all(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部语义事实 (分页)。"""
        facts = self.store.get_all_facts(user_id=user_id)
        return [
            {
                "id": f.id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "provenance": f.provenance,
                "tags": f.tags,
            }
            for f in facts[:limit]
        ]

    def get_user_profile(self, user_id: str, *, include_temporal: bool = True) -> Any:
        """从 SemanticFact 聚合用户结构化画像。"""
        from sqlmodel import Session, select

        from septmuse.models.profile import UserProfile, UserProfileValue, _classify
        from septmuse.models.semantic import SemanticFact

        engine = getattr(self.store, "engine", None)
        if engine is None:
            return UserProfile(user_id=user_id)

        with Session(engine) as session:
            stmt = select(SemanticFact).where(SemanticFact.user_id == user_id)
            facts = session.exec(stmt).all()

        predicate_facts: dict[str, list[SemanticFact]] = {}
        for f in facts:
            predicate_facts.setdefault(f.predicate, []).append(f)

        attributes: dict[str, UserProfileValue] = {}
        preferences: dict[str, UserProfileValue] = {}
        relationships: dict[str, UserProfileValue] = {}
        plans: list[UserProfileValue] = []
        raw_facts: list[dict] = []

        active_count = 0
        deleted_count = 0

        for predicate, pfacts in predicate_facts.items():
            pfacts_sorted = sorted(pfacts, key=lambda f: f.updated_at, reverse=True)
            for f in pfacts_sorted:
                if f.is_deleted:
                    deleted_count += 1
                else:
                    active_count += 1

            latest = pfacts_sorted[0]
            is_current = not latest.is_deleted

            if not include_temporal and not is_current:
                continue

            bucket = _classify(predicate)
            value_obj = UserProfileValue(
                value=latest.object,
                confidence=latest.confidence,
                updated_at=latest.updated_at.isoformat() if latest.updated_at else None,
                is_current=is_current,
                source_fact_ids=[latest.id],
            )

            if bucket == "attributes":
                attributes[predicate] = value_obj
            elif bucket == "preferences":
                preferences[predicate] = value_obj
            elif bucket == "relationships":
                relationships[predicate] = value_obj
            elif bucket == "plans":
                plans.append(value_obj)
            else:
                raw_facts.append({
                    "predicate": predicate, "value": latest.object,
                    "confidence": latest.confidence,
                    "updated_at": latest.updated_at.isoformat() if latest.updated_at else None,
                    "is_current": is_current, "fact_id": latest.id,
                })

        return UserProfile(
            user_id=user_id,
            attributes=attributes,
            preferences=preferences,
            plans=plans,
            relationships=relationships,
            raw_facts=raw_facts,
            temporal_summary={
                "active": active_count,
                "deleted": deleted_count,
                "total": active_count + deleted_count,
            },
        )
