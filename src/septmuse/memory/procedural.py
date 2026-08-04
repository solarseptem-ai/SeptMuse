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
"""V2 程序记忆子组件 — ProceduralRule CRUD + helpful/harmful 退化。

继承 LongTermMemory ABC, 委托 TypedMemoryStore。
数据模型共享 models/procedural.py 的 ProceduralRule, 不 import models/ 的操作类。

详见 docs/specs/2026-08-04-v2-memory-architecture.md §2.3 + §4。
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.memory.base import LongTermMemory
from septmuse.models.procedural import ProceduralRule
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class ProceduralMemory(LongTermMemory):
    """V2 程序记忆 — 规则 CRUD + helpful/harmful 退化。

    构造参数 (与 V1 models/procedural.py 一致):
        pm = ProceduralMemory(store=typed_store)

    与 V1 区别: 继承 LongTermMemory ABC, 实现 invalidate/get_history/get_all。
    """

    def __init__(self, store: TypedMemoryStore) -> None:
        self.store = store

    def add_rule(
        self,
        rule: str,
        *,
        user_id: str,
        namespace: str = "default",
        source_tracing: str | None = None,
        tags: list[str] | None = None,
    ) -> ProceduralRule:
        """添加规则。"""
        return self.store.add_rule(
            rule,
            user_id=user_id,
            namespace=namespace,
            source_tracing=source_tracing,
            tags=tags,
        )

    def record_outcome(self, rule_id: str, helpful: bool) -> ProceduralRule | None:
        """记录规则应用结果 (helpful/harmful 计数 + 自动退化)。"""
        return self.store.record_rule_outcome(rule_id, helpful)

    def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[ProceduralRule]:
        """获取应注入的规则 (废弃规则不返回)。"""
        return self.store.get_active_rules(user_id=user_id, namespace=namespace)

    def get_all_rules(self, *, user_id: str, include_deprecated: bool = False) -> list[ProceduralRule]:
        """列出用户全部规则。"""
        return self.store.get_all_rules(user_id=user_id, include_deprecated=include_deprecated)

    def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """编译规则为 prompt 注入文本 (仅 active 规则)。"""
        rules = self.get_active_rules(user_id=user_id, namespace=namespace)
        if not rules:
            return ""
        parts = ["<procedural_rules>"]
        for r in rules:
            parts.append(f"- {r.rule} (confidence={r.confidence:.2f})")
        parts.append("</procedural_rules>")
        return "\n".join(parts)

    # ── LongTermMemory ABC 实现 ──

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> bool:
        """标记规则不再有效 (设置 deprecated=True)。"""
        with Session(self.store.engine) as session:
            rule = session.get(ProceduralRule, memory_id)
            if not rule:
                return False
            rule.deprecated = True
            rule.touch()
            session.add(rule)
            session.commit()
            return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (审计用, 暂返回基本信息)。"""
        return [{"id": memory_id, "event": "no_history_available"}]

    def get_all(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部程序规则 (分页)。"""
        rules = self.store.get_all_rules(user_id=user_id, include_deprecated=True)
        return [
            {
                "id": r.id,
                "rule": r.rule,
                "namespace": r.namespace,
                "helpful_count": r.helpful_count,
                "harmful_count": r.harmful_count,
                "deprecated": r.deprecated,
                "confidence": r.confidence,
            }
            for r in rules[:limit]
        ]
