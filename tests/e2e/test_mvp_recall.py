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
"""E2E 阶段1 MVP — 跨会话偏好召回 (架构 §9 阶段1 + §12 验收)。

验收: 偏好记忆跨会话召回率 >= 80%。

模拟: session1 写入偏好 → session2/session3 召回, 验证持久化 + 召回率。
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory

pytestmark = pytest.mark.e2e


def _new_session(db_path: str) -> ExperimentalMemory:
    """模拟新会话 — 新 Memory 实例读同一 db 文件。"""
    return ExperimentalMemory(config=MemoryConfig(db_path=db_path), embedder=HashEmbedder())


class TestMVPRecall:
    """阶段1: 跨会话偏好召回 >= 80%。"""

    def test_preference_persists_across_sessions(self, tmp_path):
        """session1 写偏好 → session2 召回。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("Alice likes Python programming", user_id="alice")
        s1.add("Alice likes hiking on weekends", user_id="alice")
        s1.add("Alice works as backend engineer", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        results = s2.search("Alice hobbies", user_id="alice", top_k=5)
        assert len(results) > 0
        memories = [r["memory"] for r in results]
        assert any("hiking" in m for m in memories)
        s2.close()

    def test_recall_rate_above_80_percent(self, tmp_path):
        """写入 5 条偏好, 召回率 >= 80% (>=4/5)。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        preferences = [
            "Alice likes Python",
            "Alice likes coffee",
            "Alice lives in Beijing",
            "Alice works as engineer",
            "Alice speaks Chinese",
        ]
        for pref in preferences:
            s1.add(pref, user_id="alice")
        s1.close()

        s2 = _new_session(db)
        queries = [
            ("Alice programming language", "python"),
            ("Alice drinks", "coffee"),
            ("Alice lives in city", "beijing"),
            ("Alice works as professional", "engineer"),
            ("Alice speaks language", "chinese"),
        ]
        recalled = 0
        for q, expected_keyword in queries:
            results = s2.search(q, user_id="alice", top_k=3, threshold=0.01)
            for r in results:
                if expected_keyword in r["memory"].lower():
                    recalled += 1
                    break
        s2.close()

        recall_rate = recalled / len(queries)
        assert recall_rate >= 0.8, f"召回率 {recall_rate:.0%} < 80%"

    def test_user_isolation_e2e(self, tmp_path):
        """Alice 的偏好对 Bob 不可见 (跨会话用户隔离)。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("Alice likes Python", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        bob_results = s2.search("Python", user_id="bob")
        assert len(bob_results) == 0

        alice_results = s2.search("Python", user_id="alice")
        assert len(alice_results) > 0
        s2.close()

    def test_update_persists_across_sessions(self, tmp_path):
        """session1 写 → session2 update → session3 读到新内容。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        result = s1.add("old preference", user_id="alice")
        mid = result["results"][0]["id"]
        s1.close()

        s2 = _new_session(db)
        s2.update(mid, "new preference")
        s2.close()

        s3 = _new_session(db)
        item = s3.get(mid)
        assert item["memory"] == "new preference"
        history = s3.get_history(mid)
        events = [h["event"] for h in history]
        assert "ADD" in events
        assert "UPDATE" in events
        s3.close()

    def test_delete_persists_across_sessions(self, tmp_path):
        """session1 写 → session2 delete → session3 查不到。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        result = s1.add("temp memory", user_id="alice")
        mid = result["results"][0]["id"]
        s1.close()

        s2 = _new_session(db)
        s2.delete(mid)
        s2.close()

        s3 = _new_session(db)
        assert s3.get(mid) is None
        s3.close()
