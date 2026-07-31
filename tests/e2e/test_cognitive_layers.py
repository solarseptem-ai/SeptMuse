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
"""E2E 阶段2 认知分层 — 情节时序 + 程序规则退化 (架构 §9 阶段2)。

验收: 情节时序查询 + 程序规则退化流程跑通。
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory

pytestmark = pytest.mark.e2e


def _new_session(db_path: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=db_path), embedder=HashEmbedder())


class TestEpisodicTimeline:
    """情节时序查询跨会话。"""

    def test_timeline_persists_across_sessions(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add_episode("user logged in", user_id="alice", event_type="fact")
        s1.add_episode("user searched for docs", user_id="alice", event_type="fact")
        s1.add_episode("user logged out", user_id="alice", event_type="fact")
        s1.close()

        s2 = _new_session(db)
        timeline = s2.get_timeline(user_id="alice")
        assert len(timeline) == 3
        # get_timeline 返回 DESC (newest-first), reverse 为 ASC 验证时序
        contents = [e["content"] for e in reversed(timeline)]
        assert "logged in" in contents[0].lower()
        s2.close()

    def test_reasoning_episode_persists(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add_episode(
            "tried approach X",
            user_id="alice",
            event_type="reasoning",
            observation="error occurred",
            thoughts="maybe X works",
            action="tried X",
            result="X worked",
        )
        s1.close()

        s2 = _new_session(db)
        timeline = s2.get_timeline(user_id="alice", event_type="reasoning")
        assert len(timeline) == 1
        s2.close()


class TestProceduralRuleDegradation:
    """程序规则退化流程 (Cass helpful/harmful + deprecation)。"""

    def test_rule_outcome_recording(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        result = s1.add_rule("always check permissions first", user_id="alice")
        rule_id = result["id"]
        s1.close()

        s2 = _new_session(db)
        s2.record_rule_outcome(rule_id, helpful=True)
        s2.record_rule_outcome(rule_id, helpful=True)
        s2.record_rule_outcome(rule_id, helpful=False)
        s2.close()

        s3 = _new_session(db)
        active = s3.get_active_rules(user_id="alice")
        assert len(active) == 1
        s3.close()

    def test_rule_to_prompt(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add_rule("check auth before API call", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        prompt = s2.rules_to_prompt(user_id="alice")
        assert "check auth" in prompt.lower()
        s2.close()

    def test_block_persists_and_compiles(self, tmp_path):
        """工作记忆 Block 跨会话持久化 + XML 编译。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.update_block("agent-1", "human", "Name: Alice. Likes: Python")
        s1.update_block("agent-1", "persona", "I am a helpful assistant")
        s1.close()

        s2 = _new_session(db)
        wm = s2.get_working_memory("agent-1")
        xml = wm.compile_to_xml()
        assert "Alice" in xml
        assert "Python" in xml
        assert "helpful assistant" in xml
        s2.close()
