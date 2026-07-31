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
"""双时态 e2e 测试: 跨会话时态持久化 + invalidate + search_at。"""

from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.experimental import ExperimentalMemory


def test_cross_session_temporal_search(tmp_path):
    """写入 valid_at → 新实例 search_at → 正确过滤。"""
    db = str(tmp_path / "e2e_temporal.db")

    m1 = ExperimentalMemory(config=MemoryConfig(db_path=db))
    m1.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
    m1.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")

    m2 = ExperimentalMemory(config=MemoryConfig(db_path=db))
    results = m2.search_at("2024-06-01", "Alice", user_id="u1")
    assert len(results) >= 1
    assert any("Google" in r["memory"] for r in results)
    assert not any("Apple" in r["memory"] for r in results)

    results = m2.search_at("2025-06-01", "Alice", user_id="u1")
    assert any("Apple" in r["memory"] for r in results)


def test_invalidate_then_search_at(tmp_path):
    """invalidate 后 search_at 返回新事实不返回旧事实。"""
    db = str(tmp_path / "e2e_invalidate.db")
    m = ExperimentalMemory(config=MemoryConfig(db_path=db))

    r1 = m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
    mid = r1["results"][0]["id"]
    m.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")

    # 失效旧事实
    m.invalidate(mid, invalid_at="2025-01-01")

    # 2024 年应该返回 Google
    results = m.search_at("2024-06-01", "Alice", user_id="u1")
    assert any("Google" in r["memory"] for r in results)

    # 2025 年不应该返回 Google (已失效)
    results = m.search_at("2025-06-01", "Alice", user_id="u1")
    assert not any("Google" in r["memory"] for r in results)
    assert any("Apple" in r["memory"] for r in results)


def test_null_valid_at_always_returned(tmp_path):
    """valid_at=None 的记忆在 search_at 中始终返回 (向后兼容)。"""
    db = str(tmp_path / "e2e_null.db")
    m = ExperimentalMemory(config=MemoryConfig(db_path=db))
    m.add("permanent fact no time", user_id="u1")

    results = m.search_at("2024-06-01", "permanent", user_id="u1")
    assert len(results) >= 1
    results = m.search_at("2025-06-01", "permanent", user_id="u1")
    assert len(results) >= 1
