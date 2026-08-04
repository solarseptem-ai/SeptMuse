"""三元组 LLM 联合抽取。

单次 LLM 调用同时抽实体 + 关系边 (三元组), 优于分离抽取。
孤儿节点丢弃 (每实体至少有一条连接边, 否则丢弃)。
无 LLM 时 fallback 到 EntityExtractor + 简单规则生成三元组。

SeptMuse 流程:
1. LLM 模式: complete(TRIPLET_EXTRACTION_PROMPT) → {"entities": [...], "edges": [...]}
2. 丢弃孤儿实体 (没有边连接的实体)
3. 返回 list[Triplet]
4. Fallback: EntityExtractor 抽实体 → 相邻实体对 + 中间文本作为关系
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)

TRIPLET_EXTRACTION_PROMPT = """You are a knowledge graph extractor. Extract entities and relationship edges from the text in a single pass.

Output ONLY a JSON object with this exact schema:
{
  "entities": ["entity_name_1", "entity_name_2", ...],
  "edges": [
    {"source": "entity_name_1", "relation": "relation_type", "target": "entity_name_2"},
    ...
  ]
}

Rules:
- Every entity in "entities" MUST appear in at least one edge (as source or target).
- Orphan entities (no connecting edge) will be discarded.
- Use snake_case for relation types (e.g., "works_at", "located_in", "created_by").
- Keep entity names short and canonical (e.g., "Google" not "Google Inc.").
- If no relationships can be extracted, return {"entities": [], "edges": []}.

Examples:
Input: "Alice works at Google in London"
Output: {"entities": ["Alice", "Google", "London"], "edges": [{"source": "Alice", "relation": "works_at", "target": "Google"}, {"source": "Alice", "relation": "lives_in", "target": "London"}]}

Input: "Bob likes TypeScript"
Output: {"entities": ["Bob", "TypeScript"], "edges": [{"source": "Bob", "relation": "likes", "target": "TypeScript"}]}

Input: "The weather is nice today"
Output: {"entities": [], "edges": []}
"""


@dataclass
class Triplet:
    """三元组 (subject, predicate, object)。"""

    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


def _normalize_relation(between_text: str) -> str:
    """将中间文本归一化为 snake_case 关系名。"""
    cleaned = re.sub(r"[^\w\s]", "", between_text).strip().lower()
    if not cleaned:
        return "related_to"
    return re.sub(r"\s+", "_", cleaned)


def _parse_triplet_response(raw: str) -> tuple[list[str], list[dict[str, str]]]:
    """解析 LLM 输出为 (entities, edges)。"""
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("triplet_parse_failed", raw=raw[:100])
        return [], []

    if not isinstance(data, dict):
        return [], []

    entities = [str(e) for e in data.get("entities", []) if e]
    edges = []
    for edge in data.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source and relation and target:
            edges.append({"source": source, "relation": relation, "target": target})

    return entities, edges


def _drop_orphans(entities: list[str], edges: list[dict[str, str]]) -> list[str]:
    """丢弃孤儿实体 (没有边连接的实体)。"""
    connected = set()
    for edge in edges:
        connected.add(edge["source"])
        connected.add(edge["target"])
    return [e for e in entities if e in connected]


def _extract_fallback(text: str, entity_extractor: Any) -> list[Triplet]:
    """无 LLM 时用 EntityExtractor + 简单规则生成三元组。

    策略: 抽实体 → 相邻实体对 → 中间文本作为关系。
    """
    try:
        entities = entity_extractor.extract(text)
    except Exception as e:
        logger.warning("triplet_fallback_extract_failed", error=str(e))
        return []

    if len(entities) < 2:
        return []

    triplets: list[Triplet] = []
    for i in range(len(entities) - 1):
        subj = entities[i].text
        obj = entities[i + 1].text
        between = text[entities[i].end : entities[i + 1].start].strip()
        pred = _normalize_relation(between)
        triplets.append(Triplet(subject=subj, predicate=pred, object=obj))

    return triplets


class TripletExtractor:
    """三元组 LLM 联合抽取器。

    依赖注入 LLM + EntityExtractor, 便于测试 (注入 MockLLM)。
    """

    def __init__(
        self,
        llm: LLM | None = None,
        entity_extractor: Any | None = None,
    ) -> None:
        self.llm = llm
        self._entity_extractor = entity_extractor
        if self._entity_extractor is None:
            from septmuse.extraction.entity import RegexEntityExtractor

            self._entity_extractor = RegexEntityExtractor()

    def extract(self, text: str) -> list[Triplet]:
        """抽取三元组。有 LLM 走联合抽取, 无 LLM 走 fallback 规则。"""
        if not text or not text.strip():
            return []

        if self.llm is not None:
            return self._extract_with_llm(text)
        return _extract_fallback(text, self._entity_extractor)

    def _extract_with_llm(self, text: str) -> list[Triplet]:
        """单次 LLM 调用联合抽取实体 + 边。"""
        assert self.llm is not None
        raw = self.llm.complete(TRIPLET_EXTRACTION_PROMPT, f"Input:\n{text}")
        entities, edges = _parse_triplet_response(raw)

        entities = _drop_orphans(entities, edges)

        triplets = [
            Triplet(
                subject=edge["source"],
                predicate=edge["relation"],
                object=edge["target"],
            )
            for edge in edges
        ]

        logger.info("triplets_extracted", count=len(triplets), entities=len(entities))
        return triplets


def extract_triplets(
    text: str,
    llm: LLM | None = None,
    entity_extractor: Any | None = None,
) -> list[Triplet]:
    """便捷函数: 抽取三元组。

    >>> extract_triplets("Alice works at Google")
    [Triplet(subject="Alice", predicate="works_at", object="Google")]
    """
    extractor = TripletExtractor(llm=llm, entity_extractor=entity_extractor)
    return extractor.extract(text)
