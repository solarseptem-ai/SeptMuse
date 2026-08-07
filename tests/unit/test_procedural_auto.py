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
"""Memory.add(memory_type='procedural') LLM 自动生成规则测试 (对齐 mem0 _create_procedural_memory)."""
from __future__ import annotations

import os

import pytest


class _MockLLM:
    """简易 LLM mock, complete() 返回预设响应."""

    def __init__(self):
        self._response = ""

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
    llm = _MockLLM()
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


def test_procedural_prompt_exists():
    """PROCEDURAL_MEMORY_SYSTEM_PROMPT 存在且含 procedural 关键词."""
    from septmuse.prompts.extract import PROCEDURAL_MEMORY_SYSTEM_PROMPT

    assert PROCEDURAL_MEMORY_SYSTEM_PROMPT is not None
    assert "procedural" in PROCEDURAL_MEMORY_SYSTEM_PROMPT.lower()


def test_procedural_add_with_llm(memory_with_mock_llm):
    """LLM 返回规则 → Memory.add(memory_type='procedural') 存规则, event=ADD."""
    mem, llm = memory_with_mock_llm
    llm.set_response("Always validate input before processing")
    result = mem.add(
        "User asked how to handle errors. Assistant said always validate input first.",
        user_id="alice",
        memory_type="procedural",
    )
    assert result["event"] == "ADD"
    assert result["rule"] == "Always validate input before processing"
    assert result["memory_type"] == "procedural"
    assert result["id"] is not None


def test_procedural_add_llm_returns_none(memory_with_mock_llm):
    """LLM 返回 NONE → 无规则可提取, event=NOOP."""
    mem, llm = memory_with_mock_llm
    llm.set_response("NONE")
    result = mem.add("casual chat about weather", user_id="alice", memory_type="procedural")
    assert result["event"] == "NOOP"
    assert result["id"] is None
    assert result["reason"] == "no procedural knowledge found"


def test_procedural_add_no_llm_falls_back(memory_no_llm):
    """无 LLM → 原文作为规则降级存储, event=ADD."""
    result = memory_no_llm.add(
        "Always validate input before processing",
        user_id="alice",
        memory_type="procedural",
    )
    assert result["event"] == "ADD"
    assert result["rule"] == "Always validate input before processing"
