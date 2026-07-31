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
"""阶段2 类型化记忆单元测试 — 语义/情节/程序四类闭环。

固化 (架构文档 §3.2):
- SemanticFact: 三元组 + confidence + provenance + 向量检索
- EpisodicEvent: 三子类 (fact/reasoning/raw_log) + 时序查询
- ProceduralRule: helpful/harmful 退化 (Cass Playbook)
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


class TestSemanticFact:
    def test_add_fact_returns_triple(self, mem: ExperimentalMemory) -> None:
        r = mem.add_fact("alice", "likes", "python", user_id="alice")
        assert r["triple"] == {"subject": "alice", "predicate": "likes", "object": "python", "context": None}
        assert r["event"] == "ADD"

    def test_search_facts(self, mem: ExperimentalMemory) -> None:
        mem.add_fact("alice", "likes", "python", user_id="alice")
        mem.add_fact("alice", "uses", "fastapi", user_id="alice")
        hits = mem.search_facts("alice python", user_id="alice", top_k=5)
        assert len(hits) >= 1
        assert hits[0]["subject"] == "alice"

    def test_confidence_weighted(self, mem: ExperimentalMemory) -> None:
        mem.add_fact("a", "is", "high", user_id="u1", confidence=1.0)
        mem.add_fact("a", "is", "low", user_id="u1", confidence=0.1)
        hits = mem.search_facts(
            "a is",
            user_id="u1",
            top_k=5,
        )
        # high confidence 应排前 (final_score = score * confidence)
        assert hits[0]["object"] == "high"

    def test_user_isolation(self, mem: ExperimentalMemory) -> None:
        mem.add_fact("alice", "likes", "secret", user_id="alice")
        mem.add_fact("bob", "likes", "secret", user_id="bob")
        hits = mem.search_facts("secret", user_id="alice", top_k=5)
        assert all(h["subject"] == "alice" for h in hits)


class TestEpisodicEvent:
    def test_add_temporal_event(self, mem: ExperimentalMemory) -> None:
        r = mem.add_episode("event happened", user_id="u1", event_type="fact")
        assert r["event_type"] == "fact"
        assert "reference_time" in r

    def test_add_reasoning_episode(self, mem: ExperimentalMemory) -> None:
        r = mem.add_episode(
            "obs",
            user_id="u1",
            event_type="reasoning",
            observation="用户不懂递归",
            thoughts="用树屋比喻",
            action="画图",
            result="明白了",
        )
        assert r["event_type"] == "reasoning"

    def test_add_raw_log(self, mem: ExperimentalMemory) -> None:
        r = mem.add_episode("raw transcript", user_id="u1", event_type="raw_log", session_id="s1")
        assert r["event_type"] == "raw_log"

    def test_timeline(self, mem: ExperimentalMemory) -> None:
        mem.add_episode("e1", user_id="u1", event_type="fact")
        mem.add_episode(
            "e2", user_id="u1", event_type="reasoning", observation="o", thoughts="t", action="a", result="r"
        )
        mem.add_episode("e3", user_id="u1", event_type="raw_log", session_id="s1")
        tl = mem.get_timeline(user_id="u1")
        assert len(tl) == 3
        types = {e["event_type"] for e in tl}
        assert types == {"fact", "reasoning", "raw_log"}


class TestProceduralRule:
    def test_add_rule(self, mem: ExperimentalMemory) -> None:
        r = mem.add_rule("用树屋比喻讲递归", user_id="u1")
        assert r["rule"] == "用树屋比喻讲递归"
        assert r["event"] == "ADD"

    def test_record_outcome_helpful(self, mem: ExperimentalMemory) -> None:
        r = mem.add_rule("rule1", user_id="u1")
        o = mem.record_rule_outcome(r["id"], helpful=True)
        assert o["helpful_count"] == 1
        assert o["harmful_count"] == 0
        assert o["confidence"] == 1.0
        assert o["deprecated"] is False

    def test_record_outcome_harmful(self, mem: ExperimentalMemory) -> None:
        r = mem.add_rule("rule1", user_id="u1")
        o = mem.record_rule_outcome(r["id"], helpful=False)
        assert o["harmful_count"] == 1
        assert o["confidence"] == 0.0

    def test_deprecation(self, mem: ExperimentalMemory) -> None:
        """Cass 退化: harmful > helpful 且 >=3 次则 deprecated。"""
        r = mem.add_rule("bad rule", user_id="u1")
        mem.record_rule_outcome(r["id"], helpful=False)
        mem.record_rule_outcome(r["id"], helpful=False)
        o = mem.record_rule_outcome(r["id"], helpful=False)
        assert o["harmful_count"] == 3
        assert o["deprecated"] is True

    def test_deprecated_rule_not_in_active(self, mem: ExperimentalMemory) -> None:
        r = mem.add_rule("bad rule", user_id="u1")
        for _ in range(3):
            mem.record_rule_outcome(r["id"], helpful=False)
        active = mem.get_active_rules(user_id="u1")
        assert len(active) == 0

    def test_rules_to_prompt_empty(self, mem: ExperimentalMemory) -> None:
        assert mem.rules_to_prompt(user_id="u1") == ""

    def test_rules_to_prompt_has_rules(self, mem: ExperimentalMemory) -> None:
        mem.add_rule("rule1", user_id="u1")
        prompt = mem.rules_to_prompt(user_id="u1")
        assert "<procedural_rules>" in prompt
        assert "rule1" in prompt

    def test_record_nonexistent_rule(self, mem: ExperimentalMemory) -> None:
        r = mem.record_rule_outcome("rule-nonexistent", helpful=True)
        assert "error" in r
