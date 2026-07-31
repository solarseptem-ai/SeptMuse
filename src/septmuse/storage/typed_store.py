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
"""类型化记忆存储 — SQLModel engine 支持 facts/episodic/procedural 表。

对齐 solarseptem 生态 (SQLModel)。阶段1 SQLiteMemoryStore (memories 表, verbatim)
继续保留; 阶段2 类型化记忆走此 store (SemanticFact/EpisodicEvent/ProceduralRule)。

向量检索回退 numpy 余弦 (与阶段1一致, sqlite-vec 优化后续)。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import desc
from sqlmodel import Session, SQLModel, create_engine, select

from septmuse.core.logging import get_logger
from septmuse.models.block import Block, default_blocks
from septmuse.models.causal import CausalEdge
from septmuse.models.episodic import EpisodeType, EpisodicEvent
from septmuse.models.procedural import ProceduralRule
from septmuse.models.semantic import SemanticFact
from septmuse.models.strength import MemoryStrength

logger = get_logger(__name__)


def _default_db_path() -> Path:
    return Path.home() / ".septmuse" / "septmuse.db"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TypedMemoryStore:
    """类型化记忆存储 (SQLModel, 阶段2)。

    管理 SemanticFact / EpisodicEvent / ProceduralRule 三表。
    与阶段1 SQLiteMemoryStore (verbatim memories) 共存于同一 db 文件。
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _default_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})
        # create_all 建所有已 import 的 SQLModel table (facts/episodic/procedural/blocks)
        SQLModel.metadata.create_all(self.engine)
        logger.info("typed_store_ready", path=str(self.db_path))

    # ------------------------------------------------------------------
    # 语义事实 (SemanticFact)
    # ------------------------------------------------------------------

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
        embedding: list[float] | None = None,
        org_id: str = "default",
    ) -> SemanticFact:
        """添加语义事实 (对齐 LangMem Triple)。"""
        fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            object=object,
            context=context,
            user_id=user_id,
            org_id=org_id,
            confidence=confidence,
            provenance=provenance,
            tags=tags or [],
            embedding=json.dumps(embedding).encode() if embedding else None,
        )
        with Session(self.engine) as session:
            session.add(fact)
            session.commit()
            session.refresh(fact)
        logger.info(
            "fact_added",
            fact_id=fact.id,
            user_id=user_id,
            triple=f"{subject}-{predicate}->{object}",
        )
        return fact

    def update_fact(self, fact_id: str, subject: str, predicate: str, object: str) -> SemanticFact | None:
        """更新语义事实。"""
        with Session(self.engine) as session:
            fact = session.get(SemanticFact, fact_id)
            if not fact or fact.is_deleted:
                return None
            fact.subject = subject
            fact.predicate = predicate
            fact.object = object
            fact.touch()
            session.add(fact)
            session.commit()
            session.refresh(fact)
            return fact

    def soft_delete_fact(self, fact_id: str) -> bool:
        """软删除语义事实 (is_deleted=True, 借鉴 graphiti resolve_edge_contradictions)。"""
        with Session(self.engine) as session:
            fact = session.get(SemanticFact, fact_id)
            if not fact or fact.is_deleted:
                return False
            fact.is_deleted = True
            fact.touch()
            session.add(fact)
            session.commit()
            return True

    def search_facts(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索事实 (numpy 余弦, embedder 归一化则点积即余弦)。"""
        with Session(self.engine) as session:
            stmt = select(SemanticFact).where(
                SemanticFact.user_id == user_id,
                SemanticFact.is_deleted == False,  # noqa: E712
            )
            facts = session.exec(stmt).all()

        if not facts:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm > 0:
            q = q / qnorm

        results: list[dict[str, Any]] = []
        for f in facts:
            if not f.embedding:
                continue
            emb = np.array(json.loads(f.embedding), dtype=np.float32)
            score = float(np.dot(q, emb))
            if score >= threshold:
                results.append(
                    {
                        "id": f.id,
                        "subject": f.subject,
                        "predicate": f.predicate,
                        "object": f.object,
                        "context": f.context,
                        "confidence": f.confidence,
                        "provenance": f.provenance,
                        "tags": f.tags,
                        "score": score,
                    }
                )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_facts(self, *, user_id: str, include_deleted: bool = False) -> list[SemanticFact]:
        """列出用户全部事实。"""
        with Session(self.engine) as session:
            stmt = select(SemanticFact).where(SemanticFact.user_id == user_id)
            if not include_deleted:
                stmt = stmt.where(SemanticFact.is_deleted == False)  # noqa: E712
            return list(session.exec(stmt).all())

    # ------------------------------------------------------------------
    # 情节事件 (EpisodicEvent)
    # ------------------------------------------------------------------

    def add_episode(
        self,
        content: str,
        *,
        user_id: str,
        event_type: str | EpisodeType = EpisodeType.FACT,
        reference_time: datetime | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        observation: str | None = None,
        thoughts: str | None = None,
        action: str | None = None,
        result: str | None = None,
    ) -> EpisodicEvent:
        """添加情节事件 (对齐 Zep Episode + LangMem Episode)。"""
        if isinstance(event_type, EpisodeType):
            event_type = event_type.value
        event = EpisodicEvent(
            content=content,
            event_type=event_type,
            reference_time=reference_time or datetime.now(timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=result,
        )
        with Session(self.engine) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
        logger.info("episode_added", event_id=event.id, type=event_type, user_id=user_id)
        return event

    def update_episode(self, episode_id: str, content: str) -> EpisodicEvent | None:
        """更新情节事件内容。"""
        with Session(self.engine) as session:
            event = session.get(EpisodicEvent, episode_id)
            if not event or event.is_deleted:
                return None
            event.content = content
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def get_episodes(
        self,
        *,
        user_id: str,
        event_type: str | EpisodeType | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        """按时间范围查询情节 (时序检索, Zep reference_time 模式)。"""
        if isinstance(event_type, EpisodeType):
            event_type = event_type.value
        with Session(self.engine) as session:
            stmt = select(EpisodicEvent).where(
                EpisodicEvent.user_id == user_id,
                EpisodicEvent.is_deleted == False,  # noqa: E712
            )
            if event_type:
                stmt = stmt.where(EpisodicEvent.event_type == event_type)
            if since:
                stmt = stmt.where(EpisodicEvent.reference_time >= since)
            if until:
                stmt = stmt.where(EpisodicEvent.reference_time <= until)
            stmt = stmt.order_by(desc(EpisodicEvent.reference_time)).limit(limit)  # type: ignore[arg-type]  # SQLModel mypy 已知误报
            return list(session.exec(stmt).all())

    # ------------------------------------------------------------------
    # 程序规则 (ProceduralRule)
    # ------------------------------------------------------------------

    def add_rule(
        self,
        rule: str,
        *,
        user_id: str,
        namespace: str = "default",
        source_tracing: str | None = None,
        tags: list[str] | None = None,
    ) -> ProceduralRule:
        """添加程序规则 (对齐 Cass Playbook)。"""
        r = ProceduralRule(
            rule=rule,
            user_id=user_id,
            namespace=namespace,
            source_tracing=source_tracing,
            tags=tags or [],
        )
        with Session(self.engine) as session:
            session.add(r)
            session.commit()
            session.refresh(r)
        logger.info("rule_added", rule_id=r.id, user_id=user_id)
        return r

    def update_rule(self, rule_id: str, rule: str) -> ProceduralRule | None:
        """更新程序规则。"""
        with Session(self.engine) as session:
            r = session.get(ProceduralRule, rule_id)
            if not r or r.is_deleted:
                return None
            r.rule = rule
            r.touch()
            session.add(r)
            session.commit()
            session.refresh(r)
            return r

    def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[ProceduralRule]:
        """获取应注入的规则 (Cass 退化: 废弃规则不返回)。"""
        with Session(self.engine) as session:
            stmt = select(ProceduralRule).where(
                ProceduralRule.user_id == user_id,
                ProceduralRule.namespace == namespace,
                ProceduralRule.deprecated == False,  # noqa: E712
                ProceduralRule.is_deleted == False,  # noqa: E712
            )
            return list(session.exec(stmt).all())

    def record_rule_outcome(self, rule_id: str, helpful: bool) -> ProceduralRule | None:
        """记录规则应用结果 (Cass helpful/harmful 追踪 + 退化)。"""
        with Session(self.engine) as session:
            r = session.get(ProceduralRule, rule_id)
            if r is None:
                return None
            r.record_outcome(helpful)
            session.add(r)
            session.commit()
            session.refresh(r)
            return r

    def get_all_rules(self, *, user_id: str, include_deprecated: bool = False) -> list[ProceduralRule]:
        """列出用户全部规则。"""
        with Session(self.engine) as session:
            stmt = select(ProceduralRule).where(ProceduralRule.user_id == user_id)
            if not include_deprecated:
                stmt = stmt.where(ProceduralRule.is_deleted == False)  # noqa: E712
            return list(session.exec(stmt).all())

    # ------------------------------------------------------------------
    # 因果边 (CausalEdge, 架构文档 §6.1 自研)
    # ------------------------------------------------------------------

    def add_causal_edge(
        self,
        cause_event_id: str,
        effect_event_id: str,
        *,
        user_id: str,
        relation: str = "causes",
        confidence: float = 0.5,
        counterfactual_valid: bool = False,
    ) -> CausalEdge:
        """添加因果边 (架构文档 §6.1)。"""
        edge = CausalEdge(
            cause_event_id=cause_event_id,
            effect_event_id=effect_event_id,
            user_id=user_id,
            relation=relation,
            confidence=confidence,
            counterfactual_valid=counterfactual_valid,
        )
        with Session(self.engine) as session:
            session.add(edge)
            session.commit()
            session.refresh(edge)
        return edge

    def get_causes(self, event_id: str, *, user_id: str) -> list[CausalEdge]:
        """获取事件的所有因果前因 (cause → event)。"""
        with Session(self.engine) as session:
            stmt = select(CausalEdge).where(
                CausalEdge.effect_event_id == event_id,
                CausalEdge.user_id == user_id,
                CausalEdge.is_deleted == False,  # noqa: E712
            )
            return list(session.exec(stmt).all())

    def get_effects(self, event_id: str, *, user_id: str) -> list[CausalEdge]:
        """获取事件的所有因果后果 (event → effect)。"""
        with Session(self.engine) as session:
            stmt = select(CausalEdge).where(
                CausalEdge.cause_event_id == event_id,
                CausalEdge.user_id == user_id,
                CausalEdge.is_deleted == False,  # noqa: E712
            )
            return list(session.exec(stmt).all())

    # ------------------------------------------------------------------
    # 记忆强度 (MemoryStrength, 架构文档 §6.2 自研)
    # ------------------------------------------------------------------

    def get_or_create_strength(self, memory_id: str, *, user_id: str, base_value: float = 0.5) -> MemoryStrength:
        """获取或创建记忆强度记录 (架构文档 §6.2)。"""
        with Session(self.engine) as session:
            stmt = select(MemoryStrength).where(
                MemoryStrength.memory_id == memory_id,
                MemoryStrength.user_id == user_id,
                MemoryStrength.is_deleted == False,  # noqa: E712
            )
            existing = session.exec(stmt).first()
            if existing is not None:
                return existing
            strength = MemoryStrength(memory_id=memory_id, user_id=user_id, base_value=base_value)
            session.add(strength)
            session.commit()
            session.refresh(strength)
            return strength

    def update_strength(
        self,
        memory_id: str,
        *,
        user_id: str,
        strength: float,
        last_accessed: datetime | None = None,
        archived: bool | None = None,
    ) -> None:
        """更新记忆强度 (可选更新 last_accessed / archived)。"""
        with Session(self.engine) as session:
            stmt = select(MemoryStrength).where(
                MemoryStrength.memory_id == memory_id,
                MemoryStrength.user_id == user_id,
                MemoryStrength.is_deleted == False,  # noqa: E712
            )
            s = session.exec(stmt).first()
            if s is not None:
                s.strength = strength
                if last_accessed is not None:
                    s.last_accessed = last_accessed
                if archived is not None:
                    s.archived = archived
                session.add(s)
                session.commit()

    def get_all_strengths(self, *, user_id: str, include_archived: bool = False) -> list[MemoryStrength]:
        """列出用户全部记忆强度记录。"""
        with Session(self.engine) as session:
            stmt = select(MemoryStrength).where(
                MemoryStrength.user_id == user_id,
                MemoryStrength.is_deleted == False,  # noqa: E712
            )
            if not include_archived:
                stmt = stmt.where(MemoryStrength.archived == False)  # noqa: E712
            return list(session.exec(stmt).all())

    # ------------------------------------------------------------------
    # 工作记忆 Block CRUD (架构文档 §3.1.1, 对齐 Letta Block)
    # ------------------------------------------------------------------

    def get_blocks(self, agent_id: str) -> list[Block]:
        """加载 agent 的全部 block (持久化 → 内存)。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id)
            return list(session.exec(stmt).all())

    def save_block(self, block: Block) -> Block:
        """保存 block (INSERT or UPDATE, 按 id upsert)。"""
        with Session(self.engine) as session:
            existing = session.get(Block, block.id)
            if existing:
                existing.label = block.label
                existing.value = block.value
                existing.limit = block.limit
                existing.read_only = block.read_only
                existing.tags = block.tags
                existing.touch()
                session.add(existing)
            else:
                session.add(block)
            session.commit()
            session.refresh(existing or block)
            return existing or block

    def update_block_value(self, agent_id: str, label: str, value: str) -> Block | None:
        """更新 block value (对齐 WorkingMemory.update_block_value)。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return None
            block.value = value
            block.touch()
            session.add(block)
            session.commit()
            session.refresh(block)
            return block

    def delete_block(self, agent_id: str, label: str) -> bool:
        """删除 block。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return False
            session.delete(block)
            session.commit()
            return True

    def ensure_default_blocks(self, agent_id: str) -> list[Block]:
        """确保 agent 有默认 block (human + persona), 无则创建。"""
        blocks = self.get_blocks(agent_id)
        if blocks:
            return blocks
        for block in default_blocks(agent_id):
            self.save_block(block)
        return self.get_blocks(agent_id)

    def close(self) -> None:
        self.engine.dispose()
