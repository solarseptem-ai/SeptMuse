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
"""Memory.add(infer=True) 决策透传 linked_memory_ids 测试."""
from __future__ import annotations

import os

import pytest


class _MockLLM3:
    def __init__(self):
        self._response = '{"facts": []}'

    def set_response(self, r):
        self._response = r

    def complete(self, system_prompt, user_prompt):
        return self._response


@pytest.fixture
def memory_with_mock_llm(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory

    cfg = MemoryConfig(db_path=str(tmp_path / "test.db"))
    llm = _MockLLM3()
    mem = Memory(config=cfg, llm=llm)
    return mem, llm


def test_add_infer_returns_event_add(memory_with_mock_llm):
    """infer=True 决策 ADD 路径, event == ADD."""
    mem, llm = memory_with_mock_llm
    llm.set_response('{"facts":[{"text":"User likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    result = mem.add("I like Python", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"


def test_add_infer_returns_linked_memory_ids(memory_with_mock_llm):
    """infer=True 决策路径透传 linked_memory_ids 字段."""
    mem, llm = memory_with_mock_llm
    llm.set_response('{"facts":[{"text":"User likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    result = mem.add("I like Python", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert "linked_memory_ids" in result["results"][0]
    assert isinstance(result["results"][0]["linked_memory_ids"], list)


def test_add_infer_linked_nonempty_after_verbatim(memory_with_mock_llm):
    """ADD 决策双写 verbatim 后 linked_memory_ids 非空."""
    mem, llm = memory_with_mock_llm
    llm.set_response('{"facts":[{"text":"User likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    result = mem.add("I like Python", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"
    assert len(result["results"][0]["linked_memory_ids"]) >= 1


def test_add_infer_noop_has_linked_key(memory_with_mock_llm):
    """NOOP 决策也透传 linked_memory_ids (空列表)."""
    mem, llm = memory_with_mock_llm
    llm.set_response('{"facts":[{"text":"User likes Python","event":"NOOP","id":null,"confidence":1.0}]}')
    result = mem.add("I like Python", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "NOOP"
    assert "linked_memory_ids" in result["results"][0]
    assert result["results"][0]["linked_memory_ids"] == []
