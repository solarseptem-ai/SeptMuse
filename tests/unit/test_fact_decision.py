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
"""决策抽取测试 — ADD/UPDATE/DELETE/NOOP 四决策 + 置信度 + 解析容错."""
from __future__ import annotations

import os

import pytest

from septmuse.models.extract import Decision, FactExtractor


class _MockLLM:
    """测试用 Mock LLM, set_response 设定返回值, complete 返回它."""

    def __init__(self):
        self._response = '{"facts": []}'

    def set_response(self, r):
        self._response = r

    def complete(self, system_prompt, user_prompt):
        return self._response


@pytest.fixture
def mock_llm():
    return _MockLLM()


@pytest.fixture
def fact_extractor(tmp_path):
    """构建真实 FactExtractor (HashEmbedder + 真实 TypedMemoryStore + Mock LLM)."""
    import os
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.embedders.hash import HashEmbedder
    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

    embedder = HashEmbedder()
    typed_store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
    llm = _MockLLM()
    return FactExtractor(llm, embedder, typed_store, verbatim_store=None), llm, typed_store


def test_decision_dataclass():
    """Decision 基本字段."""
    d = Decision(text="Likes Rust", event="ADD")
    assert d.text == "Likes Rust"
    assert d.event == "ADD"
    assert d.id is None
    assert d.confidence == 1.0


def test_extract_with_decisions_add(fact_extractor):
    """ADD 决策."""
    ext, llm, _ = fact_extractor
    llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    decisions = ext.extract_with_decisions("I like Python", existing_memories=[])
    assert len(decisions) == 1
    assert decisions[0].text == "Likes Python"
    assert decisions[0].event == "ADD"
    assert decisions[0].confidence == 0.9


def test_extract_with_decisions_four_events(fact_extractor):
    """四决策齐全."""
    ext, llm, _ = fact_extractor
    llm.set_response(
        '{"facts":['
        '{"text":"Likes Rust","event":"ADD","id":null,"confidence":0.9},'
        '{"text":"Likes Python","event":"UPDATE","id":"mem-1","confidence":0.85},'
        '{"text":"Likes Java","event":"DELETE","id":"mem-2","confidence":0.6},'
        '{"text":"Exists","event":"NOOP","id":"mem-3","confidence":1.0}'
        "]}"
    )
    decisions = ext.extract_with_decisions(
        "msg", existing_memories=[{"id": "mem-1", "memory": "Likes Python"}]
    )
    assert len(decisions) == 4
    assert [d.event for d in decisions] == ["ADD", "UPDATE", "DELETE", "NOOP"]
    assert decisions[1].id == "mem-1"
    assert decisions[2].confidence == 0.6


def test_extract_with_decisions_parse_fallback(fact_extractor):
    """LLM 输出不合规 JSON → 降级空列表 (不阻塞)."""
    ext, llm, _ = fact_extractor
    llm.set_response("not json at all")
    decisions = ext.extract_with_decisions("I like Python", existing_memories=[])
    assert decisions == []


def test_extract_with_decisions_empty_text(fact_extractor):
    """空文本 → 空列表."""
    ext, _, _ = fact_extractor
    assert ext.extract_with_decisions("", existing_memories=[]) == []
    assert ext.extract_with_decisions("   ", existing_memories=[]) == []


def test_extract_with_decisions_invalid_event_filtered(fact_extractor):
    """非法 event 值被过滤."""
    ext, llm, _ = fact_extractor
    llm.set_response('{"facts":[{"text":"x","event":"INVALID","id":null}]}')
    decisions = ext.extract_with_decisions("msg", existing_memories=[])
    assert decisions == []  # INVALID 被过滤


# ====================================================================
# Task 2: extract_and_store 决策路由测试
# ====================================================================


class _MockLLM2:
    """决策路由测试用 Mock LLM."""

    def __init__(self):
        self._response = '{"facts": []}'

    def set_response(self, r):
        self._response = r

    def complete(self, system_prompt, user_prompt):
        return self._response


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
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    verbatim = ORMMemoryStore(engine)
    typed = TypedMemoryStore(engine=engine)
    llm = _MockLLM2()
    ext = FactExtractor(llm, embedder, typed, verbatim_store=verbatim, use_decision=True)
    return ext, llm, typed, verbatim


def test_extract_and_store_decision_add(fact_extractor_with_verbatim):
    """ADD 决策 → add_fact + verbatim add."""
    ext, llm, _, _ = fact_extractor_with_verbatim
    llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    results = ext.extract_and_store("I like Python", user_id="alice")
    assert len(results) == 1
    assert results[0]["event"] == "ADD"
    assert results[0]["id"]


def test_extract_and_store_decision_update(fact_extractor_with_verbatim):
    """UPDATE 决策 → update_fact (置信度 >=0.7)."""
    ext, llm, typed, _ = fact_extractor_with_verbatim
    # 先存一个 fact 拿 id
    from septmuse.models.extract import fact_to_triple

    s, p, o = fact_to_triple("Likes Python", "alice")
    fact = typed.add_fact(s, p, o, user_id="alice", embedding=ext.embedder.embed(f"{s} {p} {o}"))
    llm.set_response(
        f'{{"facts":[{{"text":"Likes Rust","event":"UPDATE","id":"{fact.id}","confidence":0.85}}]}}'
    )
    results = ext.extract_and_store("I like Rust now", user_id="alice")
    assert len(results) == 1
    assert results[0]["event"] == "UPDATE"
    # 验证 fact 被更新
    facts = typed.get_all_facts(user_id="alice")
    assert any(f.object == "Rust" for f in facts)


def test_extract_and_store_decision_delete(fact_extractor_with_verbatim):
    """DELETE 决策 → soft_delete_fact (置信度 >=0.7)."""
    ext, llm, typed, _ = fact_extractor_with_verbatim
    from septmuse.models.extract import fact_to_triple

    s, p, o = fact_to_triple("Likes Java", "alice")
    fact = typed.add_fact(s, p, o, user_id="alice", embedding=ext.embedder.embed(f"{s} {p} {o}"))
    llm.set_response(
        f'{{"facts":[{{"text":"Likes Java","event":"DELETE","id":"{fact.id}","confidence":0.8}}]}}'
    )
    results = ext.extract_and_store("I hate Java now", user_id="alice")
    assert len(results) == 1
    assert results[0]["event"] == "DELETE"
    # 验证软删除
    active = typed.get_all_facts(user_id="alice", include_deleted=False)
    assert all(f.id != fact.id for f in active)


def test_extract_and_store_decision_low_confidence_noop(fact_extractor_with_verbatim):
    """DELETE confidence <0.7 → 降级 NOOP (不删)."""
    ext, llm, typed, _ = fact_extractor_with_verbatim
    from septmuse.models.extract import fact_to_triple

    s, p, o = fact_to_triple("Likes Java", "alice")
    fact = typed.add_fact(s, p, o, user_id="alice", embedding=ext.embedder.embed(f"{s} {p} {o}"))
    llm.set_response(
        f'{{"facts":[{{"text":"Likes Java","event":"DELETE","id":"{fact.id}","confidence":0.5}}]}}'
    )
    results = ext.extract_and_store("msg", user_id="alice")
    assert results[0]["event"] == "NOOP"  # 降级
    # fact 仍在
    active = typed.get_all_facts(user_id="alice", include_deleted=False)
    assert any(f.id == fact.id for f in active)


def test_extract_and_store_noop(fact_extractor_with_verbatim):
    """NOOP 决策 → 跳过."""
    ext, llm, _, _ = fact_extractor_with_verbatim
    llm.set_response('{"facts":[{"text":"Exists","event":"NOOP","id":"mem-1","confidence":1.0}]}')
    results = ext.extract_and_store("msg", user_id="alice")
    assert len(results) == 1
    assert results[0]["event"] == "NOOP"


def test_extract_and_store_no_llm_legacy(fact_extractor_with_verbatim):
    """无 use_decision → 走旧 extract_facts 纯 ADD 路径."""
    ext, llm, _, _ = fact_extractor_with_verbatim
    # 创建一个 use_decision=False 的 extractor
    ext_legacy = FactExtractor(ext.llm, ext.embedder, ext.typed_store, ext.verbatim_store, use_decision=False)
    llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null}]}')
    results = ext_legacy.extract_and_store("I like Python", user_id="alice")
    # 旧路径: 解析 facts 列表, 全 ADD
    assert len(results) >= 1
    assert all(r["event"] == "ADD" for r in results)
