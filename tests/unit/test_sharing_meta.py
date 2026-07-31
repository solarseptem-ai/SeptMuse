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
"""阶段3 Batch5 共享+元认知单元测试 — user_id 共享 + metacognition 路由。

固化 (架构文档 §5.5 共享 + §5.2/§6.3 元认知):
- SharedMemoryAccessor: 跨 agent 共享查询 (mem0 user_id 模式)
- MemoryScope: 共享作用域封装
- MetaRouter: L0 路由决定查哪些命名空间
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.user_id import MemoryScope, SharedMemoryAccessor
from septmuse.meta.router import MetaRouter, RouteResult


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


# ======================================================================
# MemoryScope
# ======================================================================


class TestMemoryScope:
    def test_is_shared_no_agent(self) -> None:
        scope = MemoryScope(user_id="alice")
        assert scope.is_shared()

    def test_not_shared_with_agent(self) -> None:
        scope = MemoryScope(user_id="alice", agent_id="bot1")
        assert not scope.is_shared()

    def test_to_filter_no_agent(self) -> None:
        scope = MemoryScope(user_id="alice")
        f = scope.to_filter()
        assert f == {"user_id": "alice"}

    def test_to_filter_with_agent(self) -> None:
        scope = MemoryScope(user_id="alice", agent_id="bot1")
        f = scope.to_filter()
        assert f == {"user_id": "alice", "agent_id": "bot1"}


# ======================================================================
# SharedMemoryAccessor
# ======================================================================


class TestSharedMemoryAccessor:
    def test_list_agents_empty(self, mem: ExperimentalMemory) -> None:
        accessor = SharedMemoryAccessor(mem.store)
        assert accessor.list_agents("alice") == []

    def test_list_agents_single(self, mem: ExperimentalMemory) -> None:
        mem.add("hello", user_id="alice", agent_id="bot1")
        accessor = SharedMemoryAccessor(mem.store)
        assert accessor.list_agents("alice") == ["bot1"]

    def test_list_agents_multiple(self, mem: ExperimentalMemory) -> None:
        mem.add("msg1", user_id="alice", agent_id="bot1")
        mem.add("msg2", user_id="alice", agent_id="bot2")
        accessor = SharedMemoryAccessor(mem.store)
        agents = accessor.list_agents("alice")
        assert set(agents) == {"bot1", "bot2"}

    def test_list_users(self, mem: ExperimentalMemory) -> None:
        mem.add("msg1", user_id="alice", agent_id="bot1")
        mem.add("msg2", user_id="bob", agent_id="bot1")
        accessor = SharedMemoryAccessor(mem.store)
        users = accessor.list_users("bot1")
        assert set(users) == {"alice", "bob"}

    def test_is_cross_agent_true(self, mem: ExperimentalMemory) -> None:
        mem.add("msg1", user_id="alice", agent_id="bot1")
        mem.add("msg2", user_id="alice", agent_id="bot2")
        accessor = SharedMemoryAccessor(mem.store)
        assert accessor.is_cross_agent("alice")

    def test_is_cross_agent_false(self, mem: ExperimentalMemory) -> None:
        mem.add("msg1", user_id="alice", agent_id="bot1")
        accessor = SharedMemoryAccessor(mem.store)
        assert not accessor.is_cross_agent("alice")

    def test_is_cross_agent_shared_no_agent(self, mem: ExperimentalMemory) -> None:
        # No agent_id → shared memory
        mem.add("shared msg", user_id="alice")
        accessor = SharedMemoryAccessor(mem.store)
        assert accessor.is_cross_agent("alice")

    def test_get_shared_memories(self, mem: ExperimentalMemory) -> None:
        mem.add("msg1", user_id="alice", agent_id="bot1")
        mem.add("msg2", user_id="alice", agent_id="bot2")
        accessor = SharedMemoryAccessor(mem.store)
        memories = accessor.get_shared_memories("alice")
        assert len(memories) == 2
        assert all(m["user_id"] == "alice" for m in memories)

    def test_get_shared_memories_limit(self, mem: ExperimentalMemory) -> None:
        for i in range(5):
            mem.add(f"msg{i}", user_id="alice", agent_id=f"bot{i}")
        accessor = SharedMemoryAccessor(mem.store)
        memories = accessor.get_shared_memories("alice", limit=3)
        assert len(memories) == 3


# ======================================================================
# MetaRouter
# ======================================================================


class TestMetaRouter:
    def test_list_namespaces_default(self) -> None:
        router = MetaRouter(HashEmbedder())
        ns = router.list_namespaces()
        assert set(ns) == {"working", "semantic", "episodic", "procedural"}

    def test_route_semantic(self) -> None:
        router = MetaRouter(HashEmbedder())
        # Query shares words with semantic description
        result = router.route("preferences likes facts knowledge")
        assert "semantic" in result.namespaces
        assert not result.fallback

    def test_route_episodic(self) -> None:
        router = MetaRouter(HashEmbedder())
        result = router.route("events timeline history session")
        assert "episodic" in result.namespaces
        assert not result.fallback

    def test_route_procedural(self) -> None:
        router = MetaRouter(HashEmbedder())
        result = router.route("rules skills playbook lessons")
        assert "procedural" in result.namespaces
        assert not result.fallback

    def test_route_fallback(self) -> None:
        router = MetaRouter(HashEmbedder(), threshold=0.9)
        # No match with any namespace description
        result = router.route("xyzqwerty random stuff")
        assert result.fallback
        assert len(result.namespaces) == 4  # all namespaces

    def test_route_scores_populated(self) -> None:
        router = MetaRouter(HashEmbedder())
        result = router.route("preferences likes facts")
        assert len(result.scores) == 4
        assert all(isinstance(v, float) for v in result.scores.values())

    def test_register_namespace(self) -> None:
        router = MetaRouter(HashEmbedder())
        router.register_namespace("custom", "custom domain specific data")
        assert "custom" in router.list_namespaces()
        result = router.route("custom domain specific")
        assert "custom" in result.namespaces

    def test_route_result_dataclass(self) -> None:
        result = RouteResult(namespaces=["semantic"], scores={"semantic": 0.5})
        assert result.namespaces == ["semantic"]
        assert result.scores["semantic"] == 0.5
        assert not result.fallback

    def test_custom_namespaces(self) -> None:
        router = MetaRouter(
            HashEmbedder(),
            namespaces={"test_ns": "testing quality assurance validation"},
        )
        result = router.route("testing quality")
        assert "test_ns" in result.namespaces
        assert not result.fallback

    def test_route_working(self) -> None:
        router = MetaRouter(HashEmbedder())
        result = router.route("current context task state")
        assert "working" in result.namespaces
        assert not result.fallback
