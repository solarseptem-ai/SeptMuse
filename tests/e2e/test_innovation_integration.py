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
"""E2E 阶段4 创新增量 — 因果链 + 遗忘曲线 + 元认知 (架构 §9 阶段4 + §6)。

验收: 三项各独立的测试集 + 端到端集成。
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory

pytestmark = pytest.mark.e2e


def _new_session(db_path: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=db_path), embedder=HashEmbedder())


class TestCausalChainE2E:
    """因果链记忆端到端 (架构 §6.1)。"""

    def test_causal_edge_persists_and_queryable(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        cause = s1.add_episode("deployed v1.0", user_id="alice", event_type="fact")
        effect = s1.add_episode("error rate increased", user_id="alice", event_type="fact")
        s1.add_causal_edge(
            cause_event_id=cause["id"],
            effect_event_id=effect["id"],
            relation="causes",
            user_id="alice",
        )
        s1.close()

        s2 = _new_session(db)
        causes = s2.find_causes(effect["id"], user_id="alice")
        assert len(causes) >= 1
        effects = s2.find_effects(cause["id"], user_id="alice")
        assert len(effects) >= 1
        s2.close()

    def test_counterfactual_query(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        e1 = s1.add_episode("event A", user_id="alice", event_type="fact")
        e2 = s1.add_episode("event B", user_id="alice", event_type="fact")
        s1.add_causal_edge(e1["id"], e2["id"], relation="enables", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        result = s2.counterfactual(e1["id"], e2["id"], user_id="alice")
        assert isinstance(result, dict)
        s2.close()


class TestForgettingCurveE2E:
    """Ebbinghaus 遗忘曲线端到端 (架构 §6.2)。"""

    def test_rehearse_persists(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        result = s1.add("important fact to remember", user_id="alice")
        mid = result["results"][0]["id"]
        s1.close()

        s2 = _new_session(db)
        s2.rehearse(mid, user_id="alice")
        candidates = s2.find_rehearse_candidates(user_id="alice")
        assert isinstance(candidates, list)
        s2.close()

        s3 = _new_session(db)
        item = s3.get(mid)
        assert item is not None
        s3.close()


class TestMetacognitionE2E:
    """元认知自描述端到端 (架构 §6.3)。"""

    def test_coverage_report_e2e(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("Alice likes Python", user_id="alice")
        s1.add("Alice likes coffee", user_id="alice")
        s1.add_fact("alice", "works_as", "engineer", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        report = s2.coverage_report(user_id="alice")
        assert isinstance(report, dict)
        s2.close()

    def test_adapt_strategy_e2e(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("some memory", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        strategy = s2.adapt_strategy(user_id="alice")
        assert isinstance(strategy, dict)
        s2.close()

    def test_meta_route_e2e(self, tmp_path):
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("technical content about Python", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        route = s2.meta_route("Python programming")
        assert isinstance(route, dict)
        s2.close()


class TestSourceSyncE2E:
    """源同步器端到端 (架构 §4.1)。"""

    def test_zettel_linking_e2e(self, tmp_path):
        """add 时自动找链接 (A-MEM Zettelkasten)。"""
        db = str(tmp_path / "e2e.db")

        s1 = _new_session(db)
        s1.add("Python is a programming language", user_id="alice")
        s1.add("Python is used for web development", user_id="alice")
        s1.close()

        s2 = _new_session(db)
        results = s2.search("Python", user_id="alice")
        assert len(results) > 0
        links = s2.get_related(results[0]["id"])
        assert isinstance(links, list)
        s2.close()
