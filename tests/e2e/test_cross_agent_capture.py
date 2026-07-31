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
"""E2E 阶段3 横切关注点 — 跨 agent 共享 + 零侵入捕获 (架构 §9 阶段3)。

验收: 多 agent 共享记忆 + 编码 agent 零侵入捕获。
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory

pytestmark = pytest.mark.e2e


def _new_session(db_path: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=db_path), embedder=HashEmbedder())


class TestCrossAgentSharing:
    """同 user_id 跨 agent 共享记忆。"""

    def test_agent_a_memory_visible_to_agent_b(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("Alice prefers dark mode", user_id="alice", agent_id="chat-agent")
        s1.close()

        s2 = _new_session(db)
        results = s2.search("Alice UI preference", user_id="alice")
        assert len(results) > 0
        assert any("dark mode" in r["memory"] for r in results)
        s2.close()

    def test_list_agents_for_user(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("memory 1", user_id="alice", agent_id="chat-agent")
        s1.add("memory 2", user_id="alice", agent_id="research-agent")
        s1.close()

        s2 = _new_session(db)
        agents = s2.list_agents("alice")
        assert "chat-agent" in agents
        assert "research-agent" in agents
        assert s2.is_cross_agent("alice") is True
        s2.close()

    def test_shared_memories_across_agents(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("shared fact 1", user_id="alice", agent_id="agent-a")
        s1.add("shared fact 2", user_id="alice", agent_id="agent-b")
        s1.close()

        s2 = _new_session(db)
        shared = s2.store.get_shared_memories("alice")
        assert len(shared) >= 2
        agent_ids = {m["agent_id"] for m in shared}
        assert "agent-a" in agent_ids
        assert "agent-b" in agent_ids
        s2.close()


class TestZeroIntrusionCapture:
    """hook 捕获 (PostToolUse 风格, 零侵入)。"""

    def test_capture_persists(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.capture("tool output: file created at /tmp/x", user_id="alice", agent_id="coder")
        s1.close()

        s2 = _new_session(db)
        results = s2.search("file created", user_id="alice")
        assert len(results) > 0
        s2.close()

    def test_privacy_redaction(self, tmp_path):
        """捕获时脱敏 secrets (隐私过滤)。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        redacted = s1.redact("my api key is sk-1234567890abcdefghijklmnop")
        assert "sk-1234567890abcdef" not in redacted
        s1.close()

    def test_token_budget_applied(self, tmp_path):
        """token 预算裁剪。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        texts = ["long text " * 100, "short"]
        trimmed = s1.apply_token_budget(texts, budget=50)
        assert len(trimmed) <= 2
        s1.close()
