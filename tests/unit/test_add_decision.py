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
"""Memory.add(infer=True) 决策路由测试."""
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


@pytest.fixture
def memory_no_llm(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory

    cfg = MemoryConfig(db_path=str(tmp_path / "test.db"))
    return Memory(config=cfg)


def test_memory_add_infer_routes_add(memory_with_mock_llm):
    """Memory.add(infer=True) → 决策 ADD 路径."""
    mem, llm = memory_with_mock_llm
    llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    result = mem.add("I like Python", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"


def test_memory_add_infer_no_llm_falls_back(memory_no_llm):
    """无 LLM → infer=True 降级 verbatim 直存 (event=ADD)."""
    result = memory_no_llm.add("hello world", user_id="alice", infer=True)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"


def test_memory_add_infer_false_verbatim(memory_with_mock_llm):
    """infer=False → verbatim 直存 (不走决策)."""
    mem, _ = memory_with_mock_llm
    result = mem.add("raw text", user_id="alice", infer=False)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"
