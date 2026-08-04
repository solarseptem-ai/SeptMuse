"""冲突解决 + 实体去重 (借鉴 graphiti edge_operations resolve_edge_contradictions + node_operations 三段式去重)。

功能:
1. 矛盾事实检测: 相同 (subject, predicate) 不同 object → 冲突
2. 冲突解决: 无 LLM 用规则 (新覆盖旧), 有 LLM 判定是否真矛盾
3. 实体去重: 精确归一化 + 模糊字符串相似度 + LLM 兜底
4. 双时态失效: 复用 P2-Task 1 invalidate (verbatim memory 层)

SeptMuse 简化 (对齐 karpathy-guidelines):
- 不引入 MinHash/LSH, 用 difflib.SequenceMatcher (标准库, 零依赖)
- LLM 兜底可选 (无 LLM 时跳过第三段)
- type 升级留 TODO (当前 entity_type 不参与去重)
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    """归一化文本用于匹配 (lowercase + strip + collapse spaces)。"""
    return " ".join(text.strip().lower().split())


def _string_similarity(a: str, b: str) -> float:
    """字符串相似度 (difflib SequenceMatcher, 0-1)。"""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class ConflictResolver:
    """冲突解决 + 实体去重 (借鉴 graphiti edge_operations + node_operations)。

    用法:
        resolver = ConflictResolver(typed_store, store, llm)
        result = resolver.resolve_conflicts(user_id="u1")
        dedup_result = resolver.deduplicate_entities(user_id="u1")
    """

    def __init__(
        self,
        typed_store: TypedMemoryStore,
        store: MemoryStore | None = None,
        llm: LLM | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.typed_store = typed_store
        self.store = store
        self.llm = llm
        self.similarity_threshold = similarity_threshold

    def detect_conflicts(self, *, user_id: str) -> list[dict[str, Any]]:
        """检测矛盾事实: 相同 (subject, predicate) 不同 object。

        Returns: [{"key": (subject, predicate), "facts": [fact_dict, ...]}]
        """
        facts = self.typed_store.get_all_facts(user_id=user_id)
        groups: dict[tuple[str, str], list[Any]] = defaultdict(list)

        for fact in facts:
            key = (fact.subject, fact.predicate)
            groups[key].append(fact)

        conflicts: list[dict[str, Any]] = []
        for (subject, predicate), group_facts in groups.items():
            objects = {f.object for f in group_facts}
            if len(objects) > 1:
                conflicts.append(
                    {
                        "key": (subject, predicate),
                        "facts": [
                            {
                                "id": f.id,
                                "subject": f.subject,
                                "predicate": f.predicate,
                                "object": f.object,
                                "created_at": str(f.created_at),
                            }
                            for f in group_facts
                        ],
                    }
                )

        logger.info("conflicts_detected", user_id=user_id, count=len(conflicts))
        return conflicts

    def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """解决矛盾事实: 新事实覆盖旧事实 (验收: 矛盾事实自动失效旧事实)。

        无 LLM: 保留最新 (created_at 最大), 软删除其余。
        有 LLM: 可让 LLM 判定是否真矛盾 (TODO: LLM 判定逻辑)。

        Returns: {"conflicts_found": int, "resolved": int, "invalidated_ids": [...]}
        """
        conflicts = self.detect_conflicts(user_id=user_id)
        if not conflicts:
            return {"conflicts_found": 0, "resolved": 0, "invalidated_ids": []}

        invalidated_ids: list[str] = []
        for conflict in conflicts:
            facts = conflict["facts"]
            facts.sort(key=lambda f: f["created_at"])

            for old_fact in facts[:-1]:
                self.typed_store.update_fact(
                    old_fact["id"],
                    old_fact["subject"],
                    old_fact["predicate"],
                    old_fact["object"],
                )
                if self.typed_store.soft_delete_fact(old_fact["id"]):
                    invalidated_ids.append(old_fact["id"])

                if self.store is not None:
                    try:
                        self.store.invalidate(old_fact["id"])
                    except Exception as e:
                        logger.warning("invalidate_verbatim_failed", fact_id=old_fact["id"], error=str(e))

        logger.info("conflicts_resolved", user_id=user_id, resolved=len(invalidated_ids))
        return {
            "conflicts_found": len(conflicts),
            "resolved": len(invalidated_ids),
            "invalidated_ids": invalidated_ids,
        }

    def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """实体去重三段式 (验收: "Google"/"google"/"Google Inc" 自动合并)。

        1. 精确归一化名匹配 (恒跑)
        2. 模糊字符串相似度 (difflib, >= threshold)
        3. LLM 兜底 (可选, 无 LLM 跳过)

        Returns: {"duplicates_found": int, "merged": int, "merged_pairs": [...]}
        """
        facts = self.typed_store.get_all_facts(user_id=user_id)

        all_entities: set[str] = set()
        for fact in facts:
            all_entities.add(fact.subject)
            all_entities.add(fact.object)

        entities = list(all_entities)
        merged_pairs: list[dict[str, str]] = []
        canonical_map: dict[str, str] = {}

        for i, entity in enumerate(entities):
            if entity in canonical_map:
                continue

            normalized = _normalize(entity)
            for j in range(i + 1, len(entities)):
                other = entities[j]
                if other in canonical_map:
                    continue

                other_normalized = _normalize(other)

                if normalized == other_normalized:
                    canonical_map[other] = entity
                    merged_pairs.append({"duplicate": other, "canonical": entity, "method": "exact"})
                    continue

                sim = _string_similarity(entity, other)
                if sim >= self.similarity_threshold:
                    canonical_map[other] = entity
                    merged_pairs.append(
                        {"duplicate": other, "canonical": entity, "method": "fuzzy", "similarity": f"{sim:.3f}"}
                    )
                    continue

                if self.llm is not None and self._llm_are_duplicates(entity, other):
                    canonical_map[other] = entity
                    merged_pairs.append({"duplicate": other, "canonical": entity, "method": "llm"})

        merged_count = self._apply_canonical_map(canonical_map, user_id=user_id)

        logger.info(
            "entities_deduplicated",
            user_id=user_id,
            duplicates_found=len(merged_pairs),
            merged=merged_count,
        )

        return {
            "duplicates_found": len(merged_pairs),
            "merged": merged_count,
            "merged_pairs": merged_pairs,
        }

    def _llm_are_duplicates(self, a: str, b: str) -> bool:
        """LLM 兜底判定两个实体是否相同。"""
        if self.llm is None:
            return False
        prompt = f"Are these two entities the same? Answer ONLY 'yes' or 'no'.\nEntity 1: {a}\nEntity 2: {b}"
        try:
            response = self.llm.complete("You are an entity matching assistant.", prompt)
            return "yes" in response.strip().lower()
        except Exception as e:
            logger.warning("llm_dedup_failed", error=str(e))
            return False

    def _apply_canonical_map(self, canonical_map: dict[str, str], *, user_id: str) -> int:
        """将去重映射应用到 SemanticFact (更新 subject/object)。"""
        if not canonical_map:
            return 0

        facts = self.typed_store.get_all_facts(user_id=user_id)
        merged = 0

        for fact in facts:
            new_subject = canonical_map.get(fact.subject, fact.subject)
            new_object = canonical_map.get(fact.object, fact.object)

            if new_subject != fact.subject or new_object != fact.object:
                self.typed_store.update_fact(fact.id, new_subject, fact.predicate, new_object)
                merged += 1

        return merged
