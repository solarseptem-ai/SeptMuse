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
"""阶段3 Batch3 捕获模块单元测试 — pipeline + hooks。

固化 (架构文档 §5.1 捕获方式):
- CapturePipeline: SHA256 去重 → 隐私脱敏 → 嵌入 → 双索引
- PostToolUseHook: tool 调用结果捕获, 零侵入
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.capture.hooks import PostToolUseHook, ToolCallEvent
from septmuse.capture.pipeline import CapturePipeline
from septmuse.capture.sanitize import PrivacyFilter
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


@pytest.fixture()
def pipeline(mem: ExperimentalMemory) -> CapturePipeline:
    return CapturePipeline(
        store=mem.store,
        embedder=mem.embedder,
        typed_store=mem.typed_store,
    )


# ======================================================================
# CapturePipeline
# ======================================================================


class TestCapturePipeline:
    def test_capture_basic(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("alice likes python", user_id="alice")
        assert result.captured
        assert result.memory_id is not None
        assert result.stored_text == "alice likes python"

    def test_capture_empty_rejected(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("", user_id="alice")
        assert not result.captured
        assert "empty" in result.errors[0]

    def test_capture_whitespace_rejected(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("   ", user_id="alice")
        assert not result.captured

    def test_capture_no_user_id_rejected(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("hello", user_id="")
        assert not result.captured
        assert any("user_id" in e or "agent_id" in e for e in result.errors)

    def test_capture_dedup(self, pipeline: CapturePipeline) -> None:
        pipeline.capture("alice likes python", user_id="alice")
        result = pipeline.capture("alice likes python", user_id="alice")
        assert not result.captured
        assert result.deduped

    def test_capture_different_users_not_dedup(self, pipeline: CapturePipeline) -> None:
        pipeline.capture("hello world", user_id="alice")
        result = pipeline.capture("hello world", user_id="bob")
        assert result.captured

    def test_capture_privacy_redact(self, mem: ExperimentalMemory) -> None:
        pipeline = CapturePipeline(
            store=mem.store,
            embedder=mem.embedder,
            privacy_filter=PrivacyFilter(),
        )
        text = f"my key is sk-{'a' * 30}"
        result = pipeline.capture(text, user_id="alice")
        assert result.captured
        assert result.redacted
        assert "sk-" not in result.stored_text
        assert "[REDACTED" in result.stored_text

    def test_capture_no_redact_when_clean(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("hello world", user_id="alice")
        assert result.captured
        assert not result.redacted

    def test_capture_stored_in_store(self, pipeline: CapturePipeline, mem: ExperimentalMemory) -> None:
        result = pipeline.capture("alice likes python", user_id="alice")
        assert result.captured
        # Verify it was actually stored
        all_mem = mem.store.get_all(user_id="alice")
        assert any(m["id"] == result.memory_id for m in all_mem)

    def test_capture_metadata(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("hello", user_id="alice", metadata={"custom": "value"})
        assert result.captured
        assert result.text_hash  # SHA-256 hash should be set

    def test_capture_agent_id_alone(self, pipeline: CapturePipeline) -> None:
        result = pipeline.capture("hello", agent_id="bot1")
        assert result.captured


# ======================================================================
# PostToolUseHook
# ======================================================================


class TestPostToolUseHook:
    def test_capture_tool_output(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_output="file1.txt\nfile2.txt",
        )
        result = hook.on_post_tool_use(event, user_id="alice")
        assert result.captured
        assert "file1.txt" in result.stored_text

    def test_capture_empty_output_falls_back_to_input(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Bash",
            tool_input={"command": "git status"},
            tool_output="",
        )
        result = hook.on_post_tool_use(event, user_id="alice")
        assert result.captured
        assert "git status" in result.stored_text

    def test_capture_both_empty_uses_json(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Custom",
            tool_input={"key": "value"},
            tool_output="",
        )
        result = hook.on_post_tool_use(event, user_id="alice")
        assert result.captured
        assert "key" in result.stored_text

    def test_capture_metadata_contains_tool_name(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.txt"},
            tool_output="file contents here",
            session_id="session-123",
        )
        result = hook.on_post_tool_use(event, user_id="alice")
        assert result.captured
        # Verify metadata was stored
        all_mem = pipeline.store.get_all(user_id="alice")
        stored = next(m for m in all_mem if m["id"] == result.memory_id)
        assert stored["metadata"]["tool_name"] == "Read"
        assert stored["metadata"]["session_id"] == "session-123"

    def test_capture_dedup_across_hooks(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Bash",
            tool_input={"command": "echo hello"},
            tool_output="hello",
        )
        result1 = hook.on_post_tool_use(event, user_id="alice")
        assert result1.captured
        result2 = hook.on_post_tool_use(event, user_id="alice")
        assert not result2.captured
        assert result2.deduped

    def test_capture_send(self, pipeline: CapturePipeline) -> None:
        hook = PostToolUseHook(pipeline)
        result = hook.capture_send("manual capture text", user_id="alice")
        assert result.captured

    def test_capture_privacy_in_hook(self, mem: ExperimentalMemory) -> None:
        pipeline = CapturePipeline(
            store=mem.store,
            embedder=mem.embedder,
        )
        hook = PostToolUseHook(pipeline)
        event = ToolCallEvent(
            tool_name="Bash",
            tool_input={"command": "env"},
            tool_output=f"API_KEY=sk-{'a' * 30}",
        )
        result = hook.on_post_tool_use(event, user_id="alice")
        assert result.captured
        assert result.redacted
        assert "sk-" not in result.stored_text
