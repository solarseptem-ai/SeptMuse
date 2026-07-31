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
"""LLM 抽取流水线单元测试 (cognify, 借鉴 Cognee + mem0)。

固化 (架构文档 §3.2.2):
- MockLLM 抽取 fact 字符串 (对齐 mem0 {"facts": [...]})
- fact_to_triple 规则解析
- extract_and_store 双写 (SemanticFact + verbatim)
- facade add(infer=True) 端到端
- normalize_facts 处理变体
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.mock import MockLLM
from septmuse.models.extract import (
    fact_to_triple,
    normalize_facts,
    parse_messages,
)


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
        llm=MockLLM(),
    )


class TestParseMessages:
    def test_str(self) -> None:
        assert parse_messages("hello") == "hello"

    def test_message_list(self) -> None:
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert "user: hi" in parse_messages(msgs)
        assert "assistant: hello" in parse_messages(msgs)

    def test_empty(self) -> None:
        assert parse_messages([]) == ""


class TestNormalizeFacts:
    def test_strings(self) -> None:
        assert normalize_facts(["a", "b"]) == ["a", "b"]

    def test_dicts(self) -> None:
        assert normalize_facts([{"fact": "x"}, {"text": "y"}]) == ["x", "y"]

    def test_empty(self) -> None:
        assert normalize_facts([]) == []
        assert normalize_facts(None) == []  # type: ignore[arg-type]


class TestFactToTriple:
    def test_name(self) -> None:
        assert fact_to_triple("Name is Alice", "u1") == ("u1", "name", "Alice")

    def test_likes(self) -> None:
        assert fact_to_triple("Likes python", "u1") == ("u1", "likes", "python")

    def test_dislikes(self) -> None:
        assert fact_to_triple("Dislikes bugs", "u1") == ("u1", "dislikes", "bugs")

    def test_occupation(self) -> None:
        assert fact_to_triple("Is a engineer", "u1") == ("u1", "occupation", "engineer")

    def test_default(self) -> None:
        assert fact_to_triple("something random", "u1") == ("u1", "fact", "something random")


class TestMockLLM:
    def test_extracts_name_and_likes(self) -> None:
        llm = MockLLM()
        import json

        out = json.loads(llm.complete("sys", "Hi, my name is Alice. I like python."))
        assert "Name is Alice" in out["facts"]
        assert "Likes python" in out["facts"]

    def test_no_facts(self) -> None:
        llm = MockLLM()
        import json

        out = json.loads(llm.complete("sys", "The sky is blue."))
        assert out["facts"] == []


class TestFacadeAddInfer:
    def test_infer_true_extracts_facts(self, mem: ExperimentalMemory) -> None:
        r = mem.add("Hi, my name is Alice. I like python.", user_id="alice", infer=True)
        assert len(r["results"]) >= 2
        facts = [item["memory"] for item in r["results"]]
        assert any("Name is Alice" in f for f in facts)
        assert any("Likes python" in f for f in facts)

    def test_infer_true_stores_triples(self, mem: ExperimentalMemory) -> None:
        mem.add("my name is Bob. I like hiking.", user_id="bob", infer=True)
        facts = mem.semantic.get_all_facts(user_id="bob")
        assert len(facts) >= 2
        # 应有 name triple
        name_facts = [f for f in facts if f.predicate == "name"]
        assert len(name_facts) == 1
        assert name_facts[0].object == "Bob"

    def test_infer_true_confidence_inferred(self, mem: ExperimentalMemory) -> None:
        mem.add("my name is Alice", user_id="alice", infer=True)
        facts = mem.semantic.get_all_facts(user_id="alice")
        assert all(f.confidence == 0.7 for f in facts)
        assert all(f.provenance == "inferred" for f in facts)

    def test_infer_true_dual_write(self, mem: ExperimentalMemory) -> None:
        """cognify 双写: SemanticFact + verbatim memory 都有。"""
        mem.add("my name is Alice. I like coffee.", user_id="alice", infer=True)
        # verbatim 检索
        hits = mem.search("Alice", user_id="alice", top_k=5)
        assert len(hits) >= 1
        # semantic 检索
        facts = mem.search_facts("alice", user_id="alice", top_k=5)
        assert len(facts) >= 1

    def test_infer_false_verbatim(self, mem: ExperimentalMemory) -> None:
        r = mem.add("plain text", user_id="u1", infer=False)
        assert len(r["results"]) == 1
        assert r["results"][0]["memory"] == "plain text"
        assert r["results"][0]["event"] == "ADD"

    def test_no_llm_falls_back_verbatim(self) -> None:
        """无 LLM 时 infer=True 回退 verbatim (不崩)。"""
        m = ExperimentalMemory(config=MemoryConfig(db_path=":memory:"), embedder=HashEmbedder())
        r = m.add("my name is Alice", user_id="alice", infer=True)
        # 无 LLM, 回退 verbatim
        assert len(r["results"]) == 1
        assert "my name is Alice" in r["results"][0]["memory"]

    def test_search_after_extract(self, mem: ExperimentalMemory) -> None:
        """抽取后语义检索召回抽取的事实。"""
        mem.add("my name is Alice. I like python.", user_id="alice", infer=True)
        hits = mem.search_facts("alice name", user_id="alice", top_k=5)
        assert len(hits) >= 1
        assert any("Alice" in h["object"] for h in hits)
