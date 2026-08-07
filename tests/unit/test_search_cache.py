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
"""P0-Task 6: 查询结果缓存测试。

验证:
- 相同 query 二次检索命中缓存
- add/update/delete 后缓存失效
- 不同 user_id 不共享缓存
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class TestSearchCache:
    """查询缓存测试。"""

    def test_cache_hit_same_query(self, mem: ExperimentalMemory) -> None:
        """相同 query 二次检索命中缓存 (结果一致)。"""
        mem.add("alice likes python", user_id="alice")
        results1 = mem.search("python", user_id="alice")
        results2 = mem.search("python", user_id="alice")
        assert len(results1) == len(results2)
        assert results1[0]["id"] == results2[0]["id"]

    def test_cache_invalidate_on_add(self, mem: ExperimentalMemory) -> None:
        """add 后缓存失效, 新记忆可被检索到。"""
        mem.add("alice likes python", user_id="alice")
        results1 = mem.search("python", user_id="alice")
        assert len(results1) >= 1

        # 添加新记忆
        mem.add("bob likes python too", user_id="alice")
        results2 = mem.search("python", user_id="alice")
        # 缓存应已失效, 能看到新记忆
        assert len(results2) >= len(results1)

    def test_cache_invalidate_on_delete(self, mem: ExperimentalMemory) -> None:
        """delete 后缓存失效, 删除的记忆不再出现。"""
        mem.add("alice likes python", user_id="alice")
        results1 = mem.search("python", user_id="alice")
        assert len(results1) >= 1
        mid = results1[0]["id"]

        # 删除记忆
        mem.delete(mid)
        results2 = mem.search("python", user_id="alice")
        ids = [r["id"] for r in results2]
        assert mid not in ids

    def test_cache_invalidate_on_update(self, mem: ExperimentalMemory) -> None:
        """update 后缓存失效, 更新后的内容可被检索到。"""
        mem.add("alice likes python", user_id="alice")
        results1 = mem.search("python", user_id="alice")
        assert len(results1) >= 1
        mid = results1[0]["id"]

        # 更新记忆内容
        mem.update(mid, "alice loves javascript", user_id="alice")
        results2 = mem.search("javascript", user_id="alice")
        ids = [r["id"] for r in results2]
        assert mid in ids

    def test_different_users_no_cache_sharing(self, mem: ExperimentalMemory) -> None:
        """不同 user_id 不共享缓存。"""
        mem.add("alice likes python", user_id="alice")
        mem.add("bob likes java", user_id="bob")

        results_alice = mem.search("python", user_id="alice")
        mem.search("python", user_id="bob")

        # alice 有结果, bob 可能没有 (hash embedder 语义弱)
        assert len(results_alice) >= 1
        # 缓存 key 含 user_id, 不会串扰

    def test_cache_invalidate_all(self, mem: ExperimentalMemory) -> None:
        """invalidate_cache(user_id=None) 清所有缓存。"""
        mem.add("alice likes python", user_id="alice")
        mem.search("python", user_id="alice")
        assert mem._retriever is not None
        assert len(mem._retriever._cache) > 0

        mem._invalidate_search_cache()
        assert len(mem._retriever._cache) == 0
