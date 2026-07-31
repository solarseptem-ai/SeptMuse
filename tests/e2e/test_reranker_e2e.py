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
"""Reranker e2e 测试: 跨会话持久化 + MMR 去冗余 + explain。"""

from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.experimental import ExperimentalMemory


def test_cross_session_reranker(tmp_path):
    """写入记忆 → 新 Memory 实例 → search with reranker。"""
    db = str(tmp_path / "e2e_reranker.db")

    m1 = ExperimentalMemory(config=MemoryConfig(db_path=db))
    m1.add("Python is great", user_id="u1")
    m1.add("Java is also fine", user_id="u1")

    m2 = ExperimentalMemory(config=MemoryConfig(db_path=db))
    results = m2.search("Python", user_id="u1", reranker="noop")
    assert len(results) >= 1


def test_mmr_dedup_on_sqlite(tmp_path):
    """MMR 去冗余在真实 SQLite 上的效果。"""
    db = str(tmp_path / "e2e_mmr.db")
    m = ExperimentalMemory(config=MemoryConfig(db_path=db))

    m.add("Python programming language tutorial", user_id="u1")
    m.add("Python programming language guide", user_id="u1")
    m.add("Java programming basics", user_id="u1")

    results = m.search("Python programming", user_id="u1", reranker="mmr")
    assert len(results) >= 1


def test_explain_returns_score_details(tmp_path):
    """explain=True 返回完整 score_details。"""
    db = str(tmp_path / "e2e_explain.db")
    m = ExperimentalMemory(config=MemoryConfig(db_path=db))
    m.add("hello world from Python", user_id="u1")

    results = m.search("hello", user_id="u1", explain=True)
    assert len(results) >= 1
    meta = results[0].get("metadata", {}) or {}
    assert "score_details" in meta
    details = meta["score_details"]
    assert "vector" in details
    assert "bm25" in details
    assert "entity_boost" in details
    assert "combined" in details
