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
"""facade 集成测试 — 验证 Memory facade 暴露全部横切关注点+创新空白。

固化: Memory() 单入口可达全部功能 (架构文档 §12.4)。
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class TestFacadeCapture:
    def test_capture_basic(self, mem: ExperimentalMemory) -> None:
        result = mem.capture("alice likes python", user_id="alice")
        assert result["captured"]
        assert result["memory_id"] is not None

    def test_capture_dedup(self, mem: ExperimentalMemory) -> None:
        mem.capture("hello world", user_id="alice")
        result = mem.capture("hello world", user_id="alice")
        assert not result["captured"]
        assert result["deduped"]

    def test_capture_privacy(self, mem: ExperimentalMemory) -> None:
        result = mem.capture(f"key=sk-{'a' * 30}", user_id="alice")
        assert result["captured"]
        assert result["redacted"]


class TestFacadeAdvancedRetrieval:
    def test_search_hybrid(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        results = mem.search_hybrid("alice python", user_id="alice")
        assert len(results) >= 1
        assert "score" in results[0]

    def test_search_progressive(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        results = mem.search_progressive("alice python", user_id="alice")
        assert len(results) >= 1
        assert "memory_type" in results[0]

    def test_search_with_strength(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        results = mem.search_with_strength("alice", user_id="alice")
        assert len(results) >= 1
        assert "final_score" in results[0]
        assert "strength" in results[0]


class TestFacadeGovernance:
    def test_apply_token_budget(self, mem: ExperimentalMemory) -> None:
        texts = ["a" * 40, "b" * 40, "c" * 40]
        result = mem.apply_token_budget(texts, budget=10)
        assert len(result) <= 3

    def test_redact(self, mem: ExperimentalMemory) -> None:
        cleaned = mem.redact(f"sk-{'a' * 30}")
        assert "sk-" not in cleaned


class TestFacadeEvolution:
    def test_link_on_add(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("alice likes coding", user_id="alice")
        all_mem = mem.store.get_all(user_id="alice")
        links = mem.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], user_id="alice")
        assert isinstance(links, list)

    def test_get_related(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        all_mem = mem.store.get_all(user_id="alice")
        related = mem.get_related(all_mem[0]["id"])
        assert isinstance(related, list)

    def test_reflect_no_events(self, mem: ExperimentalMemory) -> None:
        result = mem.reflect(user_id="alice")
        assert result["proposed"] == 0

    def test_dream_empty(self, mem: ExperimentalMemory) -> None:
        result = mem.dream(user_id="alice")
        assert result["processed"] == 0


class TestFacadeSharing:
    def test_list_agents_empty(self, mem: ExperimentalMemory) -> None:
        assert mem.list_agents("alice") == []

    def test_list_agents(self, mem: ExperimentalMemory) -> None:
        mem.add("hello", user_id="alice", agent_id="bot1")
        assert mem.list_agents("alice") == ["bot1"]

    def test_is_cross_agent_false(self, mem: ExperimentalMemory) -> None:
        mem.add("hello", user_id="alice", agent_id="bot1")
        assert not mem.is_cross_agent("alice")

    def test_is_cross_agent_true(self, mem: ExperimentalMemory) -> None:
        mem.add("hello", user_id="alice", agent_id="bot1")
        mem.add("hello", user_id="alice", agent_id="bot2")
        assert mem.is_cross_agent("alice")


class TestFacadeCausal:
    def test_add_causal_edge(self, mem: ExperimentalMemory) -> None:
        e1 = mem.add_episode("deployed code", user_id="alice")
        e2 = mem.add_episode("tests passed", user_id="alice")
        result = mem.add_causal_edge(e1["id"], e2["id"], user_id="alice", relation="causes")
        assert result["relation"] == "causes"

    def test_find_causes_empty(self, mem: ExperimentalMemory) -> None:
        mem.add_episode("event", user_id="alice")
        events = mem.get_timeline(user_id="alice")
        if events:
            result = mem.find_causes(events[0]["id"], user_id="alice")
            assert isinstance(result, list)

    def test_find_effects_empty(self, mem: ExperimentalMemory) -> None:
        mem.add_episode("event", user_id="alice")
        events = mem.get_timeline(user_id="alice")
        if events:
            result = mem.find_effects(events[0]["id"], user_id="alice")
            assert isinstance(result, list)

    def test_counterfactual_no_edge(self, mem: ExperimentalMemory) -> None:
        e1 = mem.add_episode("event A", user_id="alice")
        e2 = mem.add_episode("event B", user_id="alice")
        result = mem.counterfactual(e1["id"], e2["id"], user_id="alice")
        assert "would_still_occur" in result


class TestFacadeForgetting:
    def test_rehearse(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mid = mem.store.get_all(user_id="alice")[0]["id"]
        result = mem.rehearse(mid, user_id="alice")
        assert "strength" in result
        assert result["access_count"] == 1

    def test_find_rehearse_candidates_empty(self, mem: ExperimentalMemory) -> None:
        result = mem.find_rehearse_candidates(user_id="alice")
        assert result == []


class TestFacadeMetacognition:
    def test_meta_route(self, mem: ExperimentalMemory) -> None:
        result = mem.meta_route("alice likes python")
        assert "namespaces" in result
        assert "fallback" in result
        assert len(result["namespaces"]) > 0

    def test_coverage_report_empty(self, mem: ExperimentalMemory) -> None:
        result = mem.coverage_report(user_id="alice")
        assert result["overall_score"] == 0.0
        assert len(result["namespaces"]) == 4
        assert "summary" in result

    def test_coverage_report_with_data(self, mem: ExperimentalMemory) -> None:
        for i in range(6):
            mem.add(f"memory {i}", user_id="alice")
        result = mem.coverage_report(user_id="alice")
        assert result["overall_score"] > 0
        assert "verbatim" in result["strong_areas"]

    def test_adapt_strategy_empty(self, mem: ExperimentalMemory) -> None:
        result = mem.adapt_strategy(user_id="alice")
        assert "overall_action" in result
        assert result["overall_action"] == "trigger_clarification"

    def test_adapt_strategy_strong(self, mem: ExperimentalMemory) -> None:
        for i in range(15):
            mem.add(f"memory {i}", user_id="alice")
        result = mem.adapt_strategy(user_id="alice")
        assert "recommendations" in result


class TestFacadeMethodCount:
    def test_all_methods_exposed(self, mem: ExperimentalMemory) -> None:
        """验证 facade 暴露了足够的方法 (>30)。"""
        methods = [m for m in dir(mem) if not m.startswith("_") and callable(getattr(mem, m, None))]
        assert len(methods) >= 30, f"Expected >= 30 methods, got {len(methods)}: {methods}"
