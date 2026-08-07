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
"""linked_memory_ids 跨记忆链接测试 — LLM 决策输出解析 + prompt 字段 + 端到端透传.

对齐 mem0 V3 ADDITIVE_EXTRACTION_PROMPT 的 linked_memory_ids 字段.
"""
from __future__ import annotations

import os

import pytest

from septmuse.models.extract import Decision, FactExtractor


class _MockLLM:
    """测试用 Mock LLM, set_response 设定返回值."""

    def __init__(self):
        self._response = '{"facts": []}'

    def set_response(self, r):
        self._response = r

    def complete(self, system_prompt, user_prompt):
        return self._response


# ====================================================================
# 1. _parse_decisions_response 解析 linked_memory_ids
# ====================================================================


def test_parse_decisions_with_linked_memory_ids():
    """LLM 输出含 linked_memory_ids → Decision 解析出 ["mem-3"]."""
    raw = (
        '{"facts":[{"text":"Likes sushi","event":"ADD","id":null,'
        '"confidence":0.9,"linked_memory_ids":["mem-3"]}]}'
    )
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.text == "Likes sushi"
    assert d.event == "ADD"
    assert d.confidence == 0.9
    assert d.linked_memory_ids == ["mem-3"]


def test_parse_decisions_without_linked_memory_ids():
    """LLM 输出无 linked_memory_ids → Decision.linked_memory_ids == [] (向后兼容)."""
    raw = '{"facts":[{"text":"Likes sushi","event":"ADD","id":null,"confidence":0.9}]}'
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    assert decisions[0].linked_memory_ids == []


def test_parse_decisions_empty_linked_memory_ids():
    """LLM 输出 linked_memory_ids: [] → Decision.linked_memory_ids == []."""
    raw = (
        '{"facts":[{"text":"x","event":"ADD","id":null,"confidence":1.0,'
        '"linked_memory_ids":[]}]}'
    )
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    assert decisions[0].linked_memory_ids == []


def test_parse_decisions_multiple_linked():
    """多个 linked_memory_ids 全部解析."""
    raw = (
        '{"facts":[{"text":"x","event":"ADD","id":null,"confidence":1.0,'
        '"linked_memory_ids":["a","b","c"]}]}'
    )
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    assert decisions[0].linked_memory_ids == ["a", "b", "c"]


def test_parse_decisions_linked_ids_coerced_to_str():
    """非字符串 ID 强制转 str (容错)."""
    raw = (
        '{"facts":[{"text":"x","event":"ADD","id":null,"confidence":1.0,'
        '"linked_memory_ids":[3, 7]}]}'
    )
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    assert decisions[0].linked_memory_ids == ["3", "7"]


def test_parse_decisions_linked_not_list_ignored():
    """linked_memory_ids 非列表 → 视为空 (容错)."""
    raw = (
        '{"facts":[{"text":"x","event":"ADD","id":null,"confidence":1.0,'
        '"linked_memory_ids":"mem-3"}]}'
    )
    decisions = FactExtractor._parse_decisions_response(raw)
    assert len(decisions) == 1
    assert decisions[0].linked_memory_ids == []


def test_decision_dataclass_default_empty_list():
    """Decision 默认 linked_memory_ids == [] (非 None)."""
    d = Decision(text="x", event="ADD")
    assert d.linked_memory_ids == []
    assert isinstance(d.linked_memory_ids, list)


# ====================================================================
# 2. ADDITIVE_DECISION_PROMPT 含 linked_memory_ids
# ====================================================================


def test_additive_decision_prompt_contains_linked():
    """ADDITIVE_DECISION_PROMPT 含 linked_memory_ids 字段说明."""
    from septmuse.prompts.extract import ADDITIVE_DECISION_PROMPT

    assert "linked_memory_ids" in ADDITIVE_DECISION_PROMPT


def test_additive_decision_prompt_example_has_linked():
    """示例含 linked_memory_ids 输出."""
    from septmuse.prompts.extract import ADDITIVE_DECISION_PROMPT

    assert '"linked_memory_ids":["mem-3"]' in ADDITIVE_DECISION_PROMPT


# ====================================================================
# 3. 端到端: extract_and_store ADD 路径透传 linked_memory_ids
# ====================================================================


@pytest.fixture
def fact_extractor_with_verbatim(tmp_path):
    """带 verbatim_store 的 FactExtractor (use_decision=True)."""
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from sqlalchemy import create_engine

    from septmuse.embedders.hash import HashEmbedder
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

    embedder = HashEmbedder()
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    verbatim = ORMMemoryStore(engine)
    typed = TypedMemoryStore(engine=engine)
    llm = _MockLLM()
    ext = FactExtractor(llm, embedder, typed, verbatim_store=verbatim, use_decision=True)
    return ext, llm, typed, verbatim


def test_extract_and_store_add_linked_memory_ids(fact_extractor_with_verbatim):
    """ADD 决策: LLM 输出 linked_memory_ids → 结果含 LLM 链接 + 新 vid."""
    ext, llm, _, _ = fact_extractor_with_verbatim
    llm.set_response(
        '{"facts":[{"text":"Likes sushi","event":"ADD","id":null,"confidence":0.9,'
        '"linked_memory_ids":["mem-3"]}]}'
    )
    results = ext.extract_and_store("I love sushi", user_id="alice")
    assert len(results) == 1
    r = results[0]
    assert r["event"] == "ADD"
    linked = r["linked_memory_ids"]
    assert "mem-3" in linked  # LLM 输出的跨记忆链接
    assert len(linked) == 2  # mem-3 + 新建的 verbatim id
    # 新 vid 与 mem-3 不同
    assert [v for v in linked if v != "mem-3"]


def test_extract_and_store_add_no_linked(fact_extractor_with_verbatim):
    """ADD 决策: LLM 无 linked_memory_ids → 结果只含新 vid."""
    ext, llm, _, _ = fact_extractor_with_verbatim
    llm.set_response(
        '{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}'
    )
    results = ext.extract_and_store("I like Python", user_id="alice")
    assert len(results) == 1
    r = results[0]
    assert r["event"] == "ADD"
    assert len(r["linked_memory_ids"]) == 1  # 只有新 vid
