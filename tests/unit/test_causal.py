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
"""阶段4 §6.1 因果链记忆单元测试 — CausalEdge + CausalRetriever。

固化 (架构文档 §6.1 自研):
- CausalEdge: cause→effect, relation, confidence, counterfactual_valid
- CausalRetriever: find_causes/find_effects (图遍历) + counterfactual (反事实推理)
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.models.causal import CausalEdge, CausalRelation
from septmuse.retrieval.causal import CausalPath, CausalRetriever, CounterfactualResult


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


@pytest.fixture()
def causal_setup(mem: ExperimentalMemory) -> dict[str, str]:
    """创建因果链: A causes B, B enables C, A prevents D。"""
    # 添加事件
    e_a = mem.add_episode("deployed code", user_id="alice", event_type="fact")
    e_b = mem.add_episode("tests passed", user_id="alice", event_type="fact")
    e_c = mem.add_episode("shipped to prod", user_id="alice", event_type="fact")
    e_d = mem.add_episode("rollback needed", user_id="alice", event_type="fact")

    ids = {"A": e_a["id"], "B": e_b["id"], "C": e_c["id"], "D": e_d["id"]}

    # A causes B (deployed code → tests passed)
    mem.typed_store.add_causal_edge(ids["A"], ids["B"], user_id="alice", relation="causes", confidence=0.9)
    # B enables C (tests passed → shipped to prod)
    mem.typed_store.add_causal_edge(ids["B"], ids["C"], user_id="alice", relation="enables", confidence=0.8)
    # A prevents D (deployed code → rollback not needed)
    mem.typed_store.add_causal_edge(ids["A"], ids["D"], user_id="alice", relation="prevents", confidence=0.7)

    return ids


# ======================================================================
# CausalEdge
# ======================================================================


class TestCausalEdge:
    def test_is_positive(self) -> None:
        edge = CausalEdge(
            cause_event_id="a",
            effect_event_id="b",
            relation=CausalRelation.CAUSES.value,
            user_id="alice",
        )
        assert edge.is_positive()
        assert not edge.is_negative()

    def test_is_negative(self) -> None:
        edge = CausalEdge(
            cause_event_id="a",
            effect_event_id="b",
            relation=CausalRelation.PREVENTS.value,
            user_id="alice",
        )
        assert edge.is_negative()
        assert not edge.is_positive()

    def test_default_relation(self) -> None:
        edge = CausalEdge(cause_event_id="a", effect_event_id="b", user_id="alice")
        assert edge.relation == CausalRelation.CAUSES.value

    def test_default_confidence(self) -> None:
        edge = CausalEdge(cause_event_id="a", effect_event_id="b", user_id="alice")
        assert edge.confidence == 0.5
        assert not edge.counterfactual_valid


# ======================================================================
# CausalPath
# ======================================================================


class TestCausalPath:
    def test_empty_path(self) -> None:
        path = CausalPath()
        assert path.length == 0
        assert path.confidence == 0.0
        assert path.event_ids() == []

    def test_single_edge_path(self) -> None:
        edge = CausalEdge(cause_event_id="a", effect_event_id="b", user_id="alice", confidence=0.8)
        path = CausalPath(edges=[edge])
        assert path.length == 1
        assert path.confidence == 0.8
        assert path.event_ids() == ["a", "b"]

    def test_multi_edge_path_confidence(self) -> None:
        e1 = CausalEdge(cause_event_id="a", effect_event_id="b", user_id="alice", confidence=0.9)
        e2 = CausalEdge(cause_event_id="b", effect_event_id="c", user_id="alice", confidence=0.8)
        path = CausalPath(edges=[e1, e2])
        assert path.length == 2
        assert path.confidence == pytest.approx(0.72)
        assert path.event_ids() == ["a", "b", "c"]


# ======================================================================
# CausalRetriever
# ======================================================================


class TestCausalRetriever:
    def test_find_causes_direct(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        paths = retriever.find_causes(causal_setup["B"], user_id="alice")
        # A → B (direct cause)
        assert len(paths) >= 1
        assert any(p.edges[-1].cause_event_id == causal_setup["A"] for p in paths)

    def test_find_causes_transitive(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # C's causes: B → C (direct), A → B → C (transitive)
        paths = retriever.find_causes(causal_setup["C"], user_id="alice")
        assert len(paths) >= 1
        # Should find B as direct cause
        direct_paths = [p for p in paths if p.length == 1]
        assert len(direct_paths) >= 1
        assert direct_paths[0].edges[0].cause_event_id == causal_setup["B"]

    def test_find_effects_direct(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # A's effects: A → B, A → D
        paths = retriever.find_effects(causal_setup["A"], user_id="alice")
        assert len(paths) >= 2
        effect_ids = {p.edges[-1].effect_event_id for p in paths}
        assert causal_setup["B"] in effect_ids
        assert causal_setup["D"] in effect_ids

    def test_find_effects_transitive(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # A's effects: A → B → C (transitive)
        paths = retriever.find_effects(causal_setup["A"], user_id="alice")
        # Should find transitive path A → B → C
        transitive = [p for p in paths if p.length >= 2]
        assert len(transitive) >= 1
        assert transitive[0].event_ids() == [causal_setup["A"], causal_setup["B"], causal_setup["C"]]

    def test_find_causes_empty(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # A has no causes (root cause)
        paths = retriever.find_causes(causal_setup["A"], user_id="alice")
        assert paths == []

    def test_counterfactual_no_direct_edge(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # D and B have no direct causal edge
        result = retriever.counterfactual(causal_setup["D"], causal_setup["B"], user_id="alice")
        assert result.would_still_occur
        assert result.confidence > 0.5

    def test_counterfactual_positive_no_alternative(
        self, mem: ExperimentalMemory, causal_setup: dict[str, str]
    ) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # A causes B, no alternative path to B
        result = retriever.counterfactual(causal_setup["A"], causal_setup["B"], user_id="alice")
        assert not result.would_still_occur
        assert result.confidence > 0
        assert "Positive relation" in result.reasoning

    def test_counterfactual_negative_relation(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # A prevents D → if A didn't happen, D would occur
        result = retriever.counterfactual(causal_setup["A"], causal_setup["D"], user_id="alice")
        assert result.would_still_occur
        assert "Negative relation" in result.reasoning

    def test_counterfactual_result_dataclass(self) -> None:
        result = CounterfactualResult(would_still_occur=True, confidence=0.8, reasoning="test")
        assert result.would_still_occur
        assert result.confidence == 0.8
        assert result.reasoning == "test"
        assert result.alternative_paths == []
        assert result.direct_edge is None

    def test_user_isolation(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        # Bob should not see Alice's causal edges
        paths = retriever.find_causes(causal_setup["B"], user_id="bob")
        assert paths == []

    def test_paths_sorted_by_confidence(self, mem: ExperimentalMemory, causal_setup: dict[str, str]) -> None:
        retriever = CausalRetriever(mem.typed_store)
        paths = retriever.find_effects(causal_setup["A"], user_id="alice")
        if len(paths) >= 2:
            assert paths[0].confidence >= paths[1].confidence
