#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  Unless http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""阶段5 MemOS 编排层单元测试 — registry + mem_cube + mem_os。

固化 (架构文档 §7.2 自研, 借鉴 cognee search_core_registry + MemOS):
- MemoryRegistry: 类型×形态 路由表
- MemCube: 多类型统一容器 (类型路由 + 元认知分析)
- MemOS: 端到端编排入口 (捕获→存储→检索→演化→同步)
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.memory.cube import MemCube, MemCubeConfig
from septmuse.memory.os import MemOS, MemOSConfig
from septmuse.memory.registry import MemoryRegistry, get_default_registry


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


@pytest.fixture()
def cube(mem: ExperimentalMemory) -> MemCube:
    return MemCube(mem, config=MemCubeConfig(user_id="alice"))


@pytest.fixture()
def mos(mem: ExperimentalMemory) -> MemOS:
    cube = MemCube(mem, config=MemCubeConfig(user_id="alice"))
    return MemOS(cube)


# ======================================================================
# MemoryRegistry
# ======================================================================


class TestMemoryRegistry:
    def test_register_and_get(self) -> None:
        reg = MemoryRegistry()
        handler = lambda c, m: "ok"  # noqa: E731
        reg.register("semantic", "vector", handler)
        assert reg.get("semantic", "vector") is handler

    def test_get_not_registered(self) -> None:
        reg = MemoryRegistry()
        assert reg.get("nonexistent", "form") is None

    def test_list_by_type(self) -> None:
        reg = MemoryRegistry()
        reg.register("semantic", "vector", lambda c, m: "v")
        reg.register("semantic", "graph", lambda c, m: "g")
        reg.register("episodic", "vector", lambda c, m: "e")
        entries = reg.list_by_type("semantic")
        assert len(entries) == 2

    def test_list_by_form(self) -> None:
        reg = MemoryRegistry()
        reg.register("semantic", "vector", lambda c, m: "v")
        reg.register("episodic", "vector", lambda c, m: "e")
        entries = reg.list_by_form("vector")
        assert len(entries) == 2

    def test_list_all(self) -> None:
        reg = MemoryRegistry()
        reg.register("a", "b", lambda c, m: "x")
        assert len(reg.list_all()) == 1

    def test_is_registered(self) -> None:
        reg = MemoryRegistry()
        reg.register("semantic", "vector", lambda c, m: "x")
        assert reg.is_registered("semantic", "vector")
        assert not reg.is_registered("semantic", "graph")

    def test_get_default_registry_singleton(self) -> None:
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2


# ======================================================================
# MemCube
# ======================================================================


class TestMemCube:
    def test_add_verbatim(self, cube: MemCube) -> None:
        result = cube.add("hello world", memory_type="verbatim")
        assert "results" in result

    def test_add_semantic(self, cube: MemCube) -> None:
        result = cube.add("alice", memory_type="semantic", predicate="likes", object="python")
        assert "triple" in result

    def test_add_episodic(self, cube: MemCube) -> None:
        result = cube.add("debugging session", memory_type="episodic")
        assert "id" in result

    def test_add_procedural(self, cube: MemCube) -> None:
        result = cube.add("always test", memory_type="procedural")
        assert "id" in result

    def test_search_verbatim(self, cube: MemCube) -> None:
        cube.add("alice likes python", memory_type="verbatim")
        results = cube.search("alice", memory_type="verbatim")
        assert len(results) >= 1

    def test_search_auto_route(self, cube: MemCube) -> None:
        cube.add("alice likes python", memory_type="verbatim")
        results = cube.search("alice likes python")  # auto route
        assert isinstance(results, list)

    def test_analyze(self, cube: MemCube) -> None:
        result = cube.analyze()
        assert "coverage" in result
        assert "strategy" in result

    def test_rehearse_no_candidates(self, cube: MemCube) -> None:
        result = cube.rehearse()
        assert result["rehearsed"] == 0

    def test_disabled_metacognition(self, mem: ExperimentalMemory) -> None:
        cube = MemCube(mem, config=MemCubeConfig(user_id="alice", enable_metacognition=False))
        result = cube.analyze()
        assert result["disabled"]

    def test_disabled_forgetting(self, mem: ExperimentalMemory) -> None:
        cube = MemCube(mem, config=MemCubeConfig(user_id="alice", enable_forgetting=False))
        result = cube.rehearse()
        assert result["disabled"]


# ======================================================================
# MemOS
# ======================================================================


class TestMemOS:
    def test_create_zero_config(self) -> None:
        mos = MemOS.create_zero_config(user_id="alice", embedder=HashEmbedder())
        assert mos.cube.config.user_id == "alice"

    def test_add_with_auto_evolve(self, mos: MemOS) -> None:
        result = mos.add("alice likes python", memory_type="verbatim")
        assert "results" in result

    def test_search(self, mos: MemOS) -> None:
        mos.add("alice likes python", memory_type="verbatim")
        results = mos.search("alice")
        assert isinstance(results, list)

    def test_capture(self, mos: MemOS) -> None:
        result = mos.capture("tool output")
        assert "captured" in result

    def test_analyze(self, mos: MemOS) -> None:
        result = mos.analyze()
        assert "coverage" in result

    def test_evolve(self, mos: MemOS) -> None:
        result = mos.evolve()
        assert "dream" in result
        assert "reflect" in result

    def test_rehearse(self, mos: MemOS) -> None:
        result = mos.rehearse()
        assert "rehearsed" in result

    def test_disable_auto_evolve(self, mem: ExperimentalMemory) -> None:
        cube = MemCube(mem, config=MemCubeConfig(user_id="alice"))
        mos = MemOS(cube, config=MemOSConfig(auto_evolve=False))
        result = mos.add("hello", memory_type="verbatim")
        assert "results" in result

    def test_full_loop(self, mos: MemOS) -> None:
        """端到端: 捕获→存储→检索→分析→演化→复述。"""
        # 1. 捕获
        mos.capture("alice deployed code")
        # 2. 存储
        mos.add("alice likes python", memory_type="verbatim")
        # 3. 检索
        results = mos.search("alice")
        assert len(results) >= 1
        # 4. 分析
        analysis = mos.analyze()
        assert "coverage" in analysis
        # 5. 演化
        evolve_result = mos.evolve()
        assert "dream" in evolve_result
        # 6. 复述
        rehearse_result = mos.rehearse()
        assert "rehearsed" in rehearse_result
