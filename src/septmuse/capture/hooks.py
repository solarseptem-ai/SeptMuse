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
"""PostToolUse hook — 编码 agent 零侵入记忆捕获 (架构文档 §5.1, 借鉴 Agent Memory)。

借鉴 (源码实证):
- ReMe plugins/claude_code/hooks/hooks.json: Stop 事件 hook (双 fork 后台调用 MCP)
- ReMe plugins/claude_code/hooks/auto_memory.py: stdin 读 payload → MCP auto_memory_cc
- mem0: 无 hook (显式 add)

SeptMuse 设计:
- PostToolUseHook: 接收 tool 调用结果, 提取文本, 过 CapturePipeline 存储
- capture_send: MCP transport 层的便捷方法 (对齐 mem0 capture_send)

注: ReMe 只有 Stop hook, 无 PostToolUse; 本模块实现 PostToolUse 设计 (架构文档 §5.1)。

详见 docs/specs/agent-memory-architecture.md §5.1 捕获方式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from septmuse.capture.pipeline import CapturePipeline, PipelineResult
from septmuse.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallEvent:
    """PostToolUse 事件 (对齐 Claude Code hooks PostToolUse schema)。

    字段对齐 Claude Code hook payload:
    - tool_name: 工具名 (如 "Bash", "Read", "Edit")
    - tool_input: 工具输入参数
    - tool_output: 工具输出 (文本)
    - session_id: 会话 ID
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    session_id: str | None = None


class PostToolUseHook:
    """PostToolUse 钩子 — 编码 agent 零侵入记忆捕获。

    借鉴 Agent Memory PostToolUse (架构文档 §5.1) + ReMe Stop hook 模式。
    每次 tool 调用后自动捕获输出到记忆系统。

    用法:
        hook = PostToolUseHook(pipeline)
        result = hook.on_post_tool_use(
            ToolCallEvent(tool_name="Bash", tool_input={"command": "ls"}, tool_output="file1\nfile2"),
            user_id="alice",
        )
    """

    def __init__(self, pipeline: CapturePipeline) -> None:
        self.pipeline = pipeline

    def on_post_tool_use(
        self,
        event: ToolCallEvent,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> PipelineResult:
        """处理 PostToolUse 事件, 捕获到记忆。

        从 tool_output 提取文本, 过 CapturePipeline 存储。
        """
        text = self._extract_text(event)
        if not text:
            return PipelineResult(original_text="", errors=["no text extracted"])

        metadata = {
            "tool_name": event.tool_name,
            "tool_input": json.dumps(event.tool_input, ensure_ascii=False),
            "session_id": event.session_id or "",
            "capture_source": "post_tool_use_hook",
        }
        return self.pipeline.capture(text, user_id=user_id, agent_id=agent_id, metadata=metadata)

    def _extract_text(self, event: ToolCallEvent) -> str:
        """从 ToolCallEvent 提取可记忆文本。

        策略: tool_output 是主要文本来源; 如果太短或为空, 尝试 tool_input。
        """
        text = event.tool_output.strip()
        if not text:
            # 如果输出为空, 从 input 提取 (如 command 字段)
            for key in ("command", "query", "content", "text", "prompt"):
                val = event.tool_input.get(key)
                if val and isinstance(val, str):
                    text = val.strip()
                    break
        if not text:
            # 都为空, 用 JSON 表示
            text = json.dumps(event.tool_input, ensure_ascii=False)
        return text

    def capture_send(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        tool_name: str = "manual",
    ) -> PipelineResult:
        """便捷方法: 手动捕获文本 (对齐 mem0 capture_send 模式)。

        不经过 ToolCallEvent, 直接进 pipeline。
        """
        return self.pipeline.capture(content, user_id=user_id, agent_id=agent_id, metadata={"tool_name": tool_name})
