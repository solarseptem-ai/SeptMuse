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
"""反事实因果查询 — 图遍历 + LLM 推理 (架构文档 §6.1 自研)。

14 家开源均无因果边 + 反事实查询。SeptMuse 新增:
- find_causes: 图遍历找因果前因 (直接 + 传递)
- find_effects: 图遍历找因果后果 (直接 + 传递)
- counterfactual: "若 X 未发生, Y 是否仍发生" — 图遍历找替代路径 + LLM 推理

检索排序: causal confidence × path strength
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.models.causal import CausalEdge
from septmuse.prompts.causal_extract import COUNTERFACTUAL_PROMPT, build_counterfactual_message
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)

# 最大传递深度 (防无限遍历)
MAX_TRAVERSAL_DEPTH = 5


@dataclass
class CausalPath:
    """因果路径 (从起点到终点的因果链)。"""

    edges: list[CausalEdge] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def confidence(self) -> float:
        """路径置信度 = 各边置信度的乘积。"""
        if not self.edges:
            return 0.0
        result = 1.0
        for edge in self.edges:
            result *= edge.confidence
        return result

    def event_ids(self) -> list[str]:
        """路径上的所有事件 ID (有序)。"""
        if not self.edges:
            return []
        ids = [self.edges[0].cause_event_id]
        for edge in self.edges:
            ids.append(edge.effect_event_id)
        return ids


@dataclass
class CounterfactualResult:
    """反事实查询结果。"""

    would_still_occur: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    alternative_paths: list[CausalPath] = field(default_factory=list)
    direct_edge: CausalEdge | None = None


class CausalRetriever:
    """反事实因果查询器 (架构文档 §6.1 自研)。

    用法:
        retriever = CausalRetriever(typed_store)
        causes = retriever.find_causes("epi-123", user_id="alice")
        effects = retriever.find_effects("epi-123", user_id="alice")
        result = retriever.counterfactual("epi-cause", "epi-effect", user_id="alice")
        # result.would_still_occur = True/False
    """

    def __init__(self, typed_store: TypedMemoryStore, llm: LLM | None = None) -> None:
        self.store = typed_store
        self.llm = llm

    def find_causes(self, event_id: str, *, user_id: str, max_depth: int = MAX_TRAVERSAL_DEPTH) -> list[CausalPath]:
        """找事件的因果前因 (图遍历, 直接 + 传递)。

        从 event_id 逆向遍历因果图, 找所有指向它的因果路径。
        """
        paths: list[CausalPath] = []
        self._backward_traverse(event_id, user_id, CausalPath(), paths, set(), max_depth)
        # 按置信度降序
        paths.sort(key=lambda p: p.confidence, reverse=True)
        logger.info("find_causes", event_id=event_id, paths=len(paths))
        return paths

    def find_effects(self, event_id: str, *, user_id: str, max_depth: int = MAX_TRAVERSAL_DEPTH) -> list[CausalPath]:
        """找事件的因果后果 (图遍历, 直接 + 传递)。

        从 event_id 正向遍历因果图, 找所有从它出发的因果路径。
        """
        paths: list[CausalPath] = []
        self._forward_traverse(event_id, user_id, CausalPath(), paths, set(), max_depth)
        paths.sort(key=lambda p: p.confidence, reverse=True)
        logger.info("find_effects", event_id=event_id, paths=len(paths))
        return paths

    def counterfactual(
        self,
        cause_event_id: str,
        effect_event_id: str,
        *,
        user_id: str,
    ) -> CounterfactualResult:
        """反事实查询: "若 cause 未发生, effect 是否仍发生?" (架构文档 §6.1)。

        1. 查直接因果边
        2. 查替代路径 (不经过 cause 的其他路径到 effect)
        3. 有 LLM: LLM 推理; 无 LLM: 启发式判断
        """
        result = CounterfactualResult()

        # 1. 查直接因果边
        causes = self.store.get_causes(effect_event_id, user_id=user_id)
        direct = next(
            (e for e in causes if e.cause_event_id == cause_event_id),
            None,
        )
        result.direct_edge = direct

        if direct is None:
            # 无直接因果边 → effect 不依赖 cause
            result.would_still_occur = True
            result.confidence = 0.9
            result.reasoning = "No direct causal edge found between cause and effect"
            return result

        # 2. 查替代路径 (不经过 cause 的其他路径到 effect)
        all_causes = self.store.get_causes(effect_event_id, user_id=user_id)
        alternative_causes = [e for e in all_causes if e.cause_event_id != cause_event_id]

        # 递归查替代路径
        alt_paths: list[CausalPath] = []
        for alt_edge in alternative_causes:
            alt_path = CausalPath(edges=[alt_edge])
            alt_paths.append(alt_path)
            # 递归向上找
            sub_paths = self.find_causes(alt_edge.cause_event_id, user_id=user_id, max_depth=3)
            for sp in sub_paths:
                alt_paths.append(CausalPath(edges=[*sp.edges, alt_edge]))

        result.alternative_paths = alt_paths

        # 3. 推理
        if self.llm is not None:
            return self._llm_counterfactual(cause_event_id, effect_event_id, direct, alt_paths, user_id, result)
        return self._heuristic_counterfactual(direct, alt_paths, result)

    def _llm_counterfactual(
        self,
        cause_id: str,
        effect_id: str,
        direct: CausalEdge,
        alt_paths: list[CausalPath],
        user_id: str,
        result: CounterfactualResult,
    ) -> CounterfactualResult:
        """LLM 反事实推理。"""
        cause_content = self._get_event_content(cause_id, user_id)
        effect_content = self._get_event_content(effect_id, user_id)
        alt_descriptions = [" → ".join(p.event_ids()) for p in alt_paths] if alt_paths else None

        user_msg = build_counterfactual_message(cause_content, effect_content, direct.relation, alt_descriptions)
        try:
            response = self.llm.complete(COUNTERFACTUAL_PROMPT, user_msg)  # type: ignore[union-attr]
            parsed = json.loads(response)
            result.would_still_occur = parsed.get("would_still_occur", False)
            result.confidence = float(parsed.get("confidence", 0.5))
            result.reasoning = parsed.get("reasoning", "")
        except Exception as e:
            logger.warning("llm_counterfactual_failed", error=str(e))
            return self._heuristic_counterfactual(direct, alt_paths, result)

        return result

    def _heuristic_counterfactual(
        self,
        direct: CausalEdge,
        alt_paths: list[CausalPath],
        result: CounterfactualResult,
    ) -> CounterfactualResult:
        """启发式反事实推理 (无 LLM)。"""
        if alt_paths:
            # 有替代路径 → effect 可能仍发生
            max_alt_conf = max(p.confidence for p in alt_paths)
            result.would_still_occur = True
            result.confidence = max_alt_conf * 0.7  # 降信度 (替代路径不如直接路径确定)
            result.reasoning = f"Alternative path found with confidence {max_alt_conf:.2f}"
        elif direct.is_negative():
            # 负向因果 (prevents/inhibits): cause 阻止了 effect
            # 若 cause 未发生 → effect 更可能发生
            result.would_still_occur = True
            result.confidence = direct.confidence
            result.reasoning = f"Negative relation '{direct.relation}': removing cause allows effect"
        else:
            # 正向因果 (enables/causes): cause 导致了 effect
            # 若 cause 未发生 → effect 不太可能发生
            result.would_still_occur = False
            result.confidence = direct.confidence
            result.reasoning = f"Positive relation '{direct.relation}': no alternative path, effect depends on cause"

        return result

    def _get_event_content(self, event_id: str, user_id: str) -> str:
        """获取事件内容 (用于 LLM 推理)。"""
        events = self.store.get_episodes(user_id=user_id, limit=100)
        for e in events:
            if e.id == event_id:
                return e.content or str(e.observation or "")
        return f"event-{event_id}"

    def _forward_traverse(
        self,
        event_id: str,
        user_id: str,
        current_path: CausalPath,
        all_paths: list[CausalPath],
        visited: set[str],
        depth: int,
    ) -> None:
        """正向遍历因果图 (event → effects)。"""
        if depth <= 0 or event_id in visited:
            return
        visited.add(event_id)

        effects = self.store.get_effects(event_id, user_id=user_id)
        if not effects:
            if current_path.edges:
                all_paths.append(current_path)
            return

        for edge in effects:
            new_path = CausalPath(edges=[*current_path.edges, edge])
            if edge.effect_event_id not in visited:
                all_paths.append(new_path)
                self._forward_traverse(edge.effect_event_id, user_id, new_path, all_paths, visited.copy(), depth - 1)

    def _backward_traverse(
        self,
        event_id: str,
        user_id: str,
        current_path: CausalPath,
        all_paths: list[CausalPath],
        visited: set[str],
        depth: int,
    ) -> None:
        """逆向遍历因果图 (causes → event)。"""
        if depth <= 0 or event_id in visited:
            return
        visited.add(event_id)

        causes = self.store.get_causes(event_id, user_id=user_id)
        if not causes:
            if current_path.edges:
                all_paths.append(current_path)
            return

        for edge in causes:
            new_path = CausalPath(edges=[edge, *current_path.edges])
            if edge.cause_event_id not in visited:
                all_paths.append(new_path)
                self._backward_traverse(edge.cause_event_id, user_id, new_path, all_paths, visited.copy(), depth - 1)
