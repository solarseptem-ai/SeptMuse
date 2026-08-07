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
"""FactExtractor 上下文窗口 (last_k_messages) + prompt 构建测试."""
from __future__ import annotations


class _MockEvent:
    def __init__(self, content: str):
        self.content = content


class _MockEpisodicStore:
    """模拟 EpisodicMemory.get_timeline 行为."""

    def __init__(self, events: list[_MockEvent] | None = None):
        self._events = events or [_MockEvent("hello"), _MockEvent("world")]

    def get_timeline(self, *, user_id: str, limit: int = 5):
        return self._events[:limit]


def _make_extractor(episodic_store=None):
    """创建 FactExtractor (最小化依赖, 仅测上下文窗口逻辑)."""
    from septmuse.models.extract import FactExtractor

    return FactExtractor(
        llm=None,
        embedder=None,
        typed_store=None,
        verbatim_store=None,
        use_decision=True,
        episodic_store=episodic_store,
    )


class TestGetLastKMessages:
    def test_no_episodic_store_returns_empty(self):
        """无 episodic_store → 空列表 (降级)."""
        extractor = _make_extractor(episodic_store=None)
        result = extractor._get_last_k_messages("alice")
        assert result == []

    def test_with_episodic_store_returns_nonempty(self):
        """有 episodic_store → 返回非空上下文列表."""
        mock_store = _MockEpisodicStore()
        extractor = _make_extractor(episodic_store=mock_store)
        result = extractor._get_last_k_messages("alice")
        assert len(result) == 2
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "world"

    def test_role_is_assistant(self):
        """返回的 role 字段为 assistant."""
        mock_store = _MockEpisodicStore()
        extractor = _make_extractor(episodic_store=mock_store)
        result = extractor._get_last_k_messages("alice")
        assert all(m["role"] == "assistant" for m in result)

    def test_limit_respected(self):
        """limit 参数限制返回条数."""
        mock_store = _MockEpisodicStore([_MockEvent(f"msg{i}") for i in range(10)])
        extractor = _make_extractor(episodic_store=mock_store)
        result = extractor._get_last_k_messages("alice", limit=3)
        assert len(result) == 3

    def test_exception_returns_empty(self):
        """episodic_store 抛异常 → 降级空列表 (不阻塞)."""

        class _BadStore:
            def get_timeline(self, *, user_id, limit=5):
                raise RuntimeError("db gone")

        extractor = _make_extractor(episodic_store=_BadStore())
        result = extractor._get_last_k_messages("alice")
        assert result == []


class TestBuildExtractionUserPrompt:
    def test_no_last_k_omits_section(self):
        """无 last_k_messages → 不含 Last k Messages 段落."""
        from septmuse.prompts.extract import build_extraction_user_prompt

        prompt = build_extraction_user_prompt("hello world", None)
        assert "Last k Messages" not in prompt
        assert "## New Messages\nhello world" in prompt

    def test_with_last_k_includes_section(self):
        """有 last_k_messages → 含 Last k Messages 段落."""
        from septmuse.prompts.extract import build_extraction_user_prompt

        last_k = [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello back"},
        ]
        prompt = build_extraction_user_prompt("new msg", None, last_k_messages=last_k)
        assert "## Last k Messages" in prompt
        assert "hi there" in prompt
        assert "hello back" in prompt

    def test_with_last_k_role_and_content(self):
        """last_k 段落含 role: content 格式."""
        from septmuse.prompts.extract import build_extraction_user_prompt

        last_k = [{"role": "user", "content": "test message"}]
        prompt = build_extraction_user_prompt("new", None, last_k_messages=last_k)
        assert "user: test message" in prompt

    def test_empty_last_k_omits_section(self):
        """空 last_k_messages 列表 → 不含段落 (falsy)."""
        from septmuse.prompts.extract import build_extraction_user_prompt

        prompt = build_extraction_user_prompt("new", None, last_k_messages=[])
        assert "Last k Messages" not in prompt

    def test_last_k_before_existing_memories(self):
        """last_k 段落出现在 existing memories 之前 (上下文优先)."""
        from septmuse.prompts.extract import build_extraction_user_prompt

        last_k = [{"role": "user", "content": "ctx line"}]
        existing = [{"id": "mem-1", "memory": "old fact"}]
        prompt = build_extraction_user_prompt("new", existing, last_k_messages=last_k)
        idx_last_k = prompt.index("## Last k Messages")
        idx_existing = prompt.index("## Existing Memories")
        assert idx_last_k < idx_existing
