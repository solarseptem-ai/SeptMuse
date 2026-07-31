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
"""遗忘曲线检索 — 强度加权排序 + 主动复述 (架构文档 §6.2 自研)。

14 家开源均无强度衰减 + 主动复述。SeptMuse 新增:
- apply_strength: final_score = relevance × decayed_strength, 低强度记忆下沉
- find_rehearse_candidates: 扫描 strength < 0.3 且 base_value > 0.7 的记忆
- rehearse: 主动复述 (strength 回升 + access_count+1)
- archive: strength < 0.1 → 归档冷存储
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.models.strength import MemoryStrength
from septmuse.storage.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class StrengthWeightedResult:
    """强度加权后的检索结果。"""

    id: str
    memory: str
    relevance: float  # 原始相关度
    strength: float  # 当前衰减后强度
    final_score: float  # relevance × strength
    metadata: dict[str, Any] = field(default_factory=dict)


class ForgettingRetriever:
    """遗忘曲线检索器 (架构文档 §6.2 自研)。

    用法:
        retriever = ForgettingRetriever(typed_store)
        weighted = retriever.apply_strength(results, user_id="alice")
        candidates = retriever.find_rehearse_candidates(user_id="alice")
        retriever.rehearse_batch([m.memory_id for m in candidates], user_id="alice")
    """

    def __init__(self, typed_store: TypedMemoryStore) -> None:
        self.store = typed_store

    def apply_strength(
        self,
        results: list[dict[str, Any]],
        *,
        user_id: str,
        now: datetime | None = None,
    ) -> list[StrengthWeightedResult]:
        """对检索结果应用强度加权 (final_score = relevance × strength)。

        对每条结果:
        1. 查 MemoryStrength (不存在则创建, 默认 base_value=0.5)
        2. 计算 decayed_strength
        3. final_score = relevance × decayed_strength
        4. 按 final_score 降序排序

        归档记忆 (archived=True) 不参与, 被过滤掉。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        weighted: list[StrengthWeightedResult] = []
        for r in results:
            mid = r.get("id", "")
            if not mid:
                continue
            strength = self.store.get_or_create_strength(mid, user_id=user_id)
            if strength.archived:
                continue  # 归档记忆不参与默认检索

            decayed = strength.decay(now)
            relevance = r.get("score", 0.0)
            final = relevance * decayed

            weighted.append(
                StrengthWeightedResult(
                    id=mid,
                    memory=r.get("memory", ""),
                    relevance=relevance,
                    strength=decayed,
                    final_score=final,
                    metadata=r.get("metadata", {}),
                )
            )

        weighted.sort(key=lambda x: x.final_score, reverse=True)
        logger.info("apply_strength_done", input_count=len(results), output_count=len(weighted))
        return weighted

    def find_rehearse_candidates(
        self,
        *,
        user_id: str,
        now: datetime | None = None,
    ) -> list[MemoryStrength]:
        """找需要复述的记忆 (strength < 0.3 且 base_value > 0.7, 架构文档 §6.2)。

        agent idle / Dream 阶段调用, 主动复述高价值低强度记忆。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        all_strengths = self.store.get_all_strengths(user_id=user_id, include_archived=False)
        candidates = [s for s in all_strengths if s.should_rehearse(now)]
        logger.info("rehearse_candidates", user_id=user_id, candidates=len(candidates))
        return candidates

    def rehearse(self, memory_id: str, *, user_id: str, now: datetime | None = None) -> MemoryStrength | None:
        """复述单条记忆 (strength 回升 + access_count+1)。"""
        if now is None:
            now = datetime.now(timezone.utc)

        strength = self.store.get_or_create_strength(memory_id, user_id=user_id)
        strength.rehearse(now)
        self.store.update_strength(
            memory_id,
            user_id=user_id,
            strength=strength.strength,
            last_accessed=strength.last_accessed,
            archived=strength.archived,
        )
        logger.info(
            "rehearse_done", memory_id=memory_id, strength=strength.strength, access_count=strength.access_count
        )
        return strength

    def rehearse_batch(self, memory_ids: list[str], *, user_id: str, now: datetime | None = None) -> int:
        """批量复述。"""
        count = 0
        for mid in memory_ids:
            if self.rehearse(mid, user_id=user_id, now=now) is not None:
                count += 1
        logger.info("rehearse_batch_done", count=count)
        return count

    def archive_stale(
        self,
        *,
        user_id: str,
        now: datetime | None = None,
    ) -> list[str]:
        """归档陈旧记忆 (strength < 0.1, 架构文档 §6.2 退化)。

        归档后的记忆不参与默认检索 (冷存储), 但仍可手动召回。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        all_strengths = self.store.get_all_strengths(user_id=user_id, include_archived=False)
        to_archive = [s for s in all_strengths if s.should_archive(now)]
        archived_ids: list[str] = []
        for s in to_archive:
            archived_ids.append(s.memory_id)
            self.store.update_strength(
                s.memory_id,
                user_id=user_id,
                strength=s.decay(now),
                archived=True,
            )

        logger.info("archive_stale_done", user_id=user_id, archived=len(archived_ids))
        return archived_ids
