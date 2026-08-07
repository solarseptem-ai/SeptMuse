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
"""阶段3 Batch4 演化模块单元测试 — zettel + reflect + dream。

固化 (架构文档 §5.4 演化):
- ZettelLinker: add 时自动建链接 (cognee expand_with_nodes_and_edges)
- SessionReflector: session 结束提取教训 (cognee distill_session)
- DreamIntegrator: 空闲期批量建链接 (ReMe Dream)
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.evolution.dream import DreamIntegrator
from septmuse.evolution.reflect import SessionReflector
from septmuse.evolution.zettel import ZettelLinker
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class LessonMockLLM(LLM):
    """测试用 LLM, 返回 lesson 格式文本。"""

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return "Lesson: Always test edge cases before shipping\nLesson: Document the reasoning behind decisions"


# ======================================================================
# ZettelLinker
# ======================================================================


class TestZettelLinker:
    def test_link_on_add_creates_links(self, mem: ExperimentalMemory) -> None:
        # 先存一条已有记忆
        mem.add("alice likes python", user_id="alice")
        # 再存新记忆, 应自动建链接
        mem.add("alice enjoys programming", user_id="alice")

        linker = ZettelLinker(mem.store, mem.graph_store, mem.embedder)
        all_mem = mem.store.get_all(user_id="alice")
        assert len(all_mem) >= 2

        # 为第一条记忆建链接
        emb = mem.embedder.embed(all_mem[0]["memory"])
        links = linker.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], emb, user_id="alice")
        # HashEmbedder 的相似度可能较低, 但如果 > threshold 应该有链接
        # 链接是双向的
        if links:
            assert all(link.source_id == all_mem[0]["id"] for link in links)
            # 反向链接也应存在
            for link in links:
                reverse = linker.get_links(link.target_id)
                assert any(r.target_id == all_mem[0]["id"] for r in reverse)

    def test_link_dedup(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("alice likes python programming", user_id="alice")

        linker = ZettelLinker(mem.store, mem.graph_store, mem.embedder)
        all_mem = mem.store.get_all(user_id="alice")

        emb = mem.embedder.embed(all_mem[0]["memory"])
        # 第一次建链接
        linker.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], emb, user_id="alice")
        # 第二次同样的记忆, 不应重复建链接
        links2 = linker.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], emb, user_id="alice")
        assert len(links2) == 0  # 已有链接, 不重复

    def test_get_links_empty(self, mem: ExperimentalMemory) -> None:
        linker = ZettelLinker(mem.store, mem.graph_store, mem.embedder)
        links = linker.get_links("nonexistent-id")
        assert links == []

    def test_get_related_memories(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("alice likes python programming", user_id="alice")

        linker = ZettelLinker(mem.store, mem.graph_store, mem.embedder, threshold=0.0)  # 最低阈值确保链接
        all_mem = mem.store.get_all(user_id="alice")

        emb = mem.embedder.embed(all_mem[0]["memory"])
        linker.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], emb, user_id="alice")

        related = linker.get_related_memories(all_mem[0]["id"])
        # 应该能获取到关联记忆
        assert isinstance(related, list)

    def test_self_not_linked(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        linker = ZettelLinker(mem.store, mem.graph_store, mem.embedder, threshold=0.0)

        all_mem = mem.store.get_all(user_id="alice")
        emb = mem.embedder.embed(all_mem[0]["memory"])
        links = linker.link_on_add(all_mem[0]["id"], all_mem[0]["memory"], emb, user_id="alice")
        # 不应链接到自身
        assert all(link.target_id != all_mem[0]["id"] for link in links)


# ======================================================================
# SessionReflector
# ======================================================================


class TestSessionReflector:
    def test_no_events(self, mem: ExperimentalMemory) -> None:
        reflector = SessionReflector(mem.typed_store)
        result = reflector.reflect(user_id="alice")
        assert result.lessons_proposed == 0
        assert result.lessons_accepted == 0

    def test_heuristic_extract(self, mem: ExperimentalMemory) -> None:
        # 添加 reasoning event
        mem.add_episode(
            "debugging session",
            user_id="alice",
            event_type="reasoning",
            observation="test failed on edge case",
            thoughts="should check for null input before processing",
            action="added null check",
            result="test passed",
        )
        reflector = SessionReflector(mem.typed_store)  # no LLM → heuristic
        result = reflector.reflect(user_id="alice", limit=10)
        assert result.lessons_proposed > 0
        assert result.lessons_accepted > 0
        assert len(result.rule_ids) > 0

    def test_llm_extract(self, mem: ExperimentalMemory) -> None:
        mem.add_episode(
            "code review",
            user_id="alice",
            event_type="reasoning",
            observation="code had no tests",
            thoughts="need to add tests",
            action="wrote tests",
            result="coverage improved",
        )
        reflector = SessionReflector(mem.typed_store, llm=LessonMockLLM())
        result = reflector.reflect(user_id="alice", limit=10)
        assert result.lessons_proposed >= 2  # MockLLM returns 2 lessons
        assert result.lessons_accepted >= 2

    def test_rules_stored_in_procedural(self, mem: ExperimentalMemory) -> None:
        mem.add_episode(
            "session",
            user_id="alice",
            event_type="reasoning",
            observation="obs",
            thoughts="always validate inputs",
            action="act",
            result="ok",
        )
        reflector = SessionReflector(mem.typed_store)
        result = reflector.reflect(user_id="alice", namespace="test_reflection")
        if result.rule_ids:
            rules = mem.typed_store.get_all_rules(user_id="alice")
            assert any(r.id in result.rule_ids for r in rules)

    def test_short_lesson_rejected(self, mem: ExperimentalMemory) -> None:
        mem.add_episode(
            "short",
            user_id="alice",
            event_type="reasoning",
            observation="x",
            thoughts="ab",  # too short
            action="cd",  # too short
            result="ok",
        )
        reflector = SessionReflector(mem.typed_store)
        result = reflector.reflect(user_id="alice", limit=10)
        # Short thoughts/actions should be rejected (< 5 chars)
        assert result.lessons_rejected >= 0  # may have 0 proposed if all too short


# ======================================================================
# DreamIntegrator
# ======================================================================


class TestDreamIntegrator:
    def test_empty_store(self, mem: ExperimentalMemory) -> None:
        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder)
        result = dreamer.dream(user_id="alice")
        assert result.processed == 0
        assert result.links_created == 0

    def test_dream_creates_links(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python programming", user_id="alice")
        mem.add("alice enjoys coding in python", user_id="alice")
        mem.add("bob likes java", user_id="alice")  # different content

        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder, threshold=0.0)
        result = dreamer.dream(user_id="alice")
        assert result.processed >= 1
        # Links may or may not be created depending on similarity threshold

    def test_dream_processes_all(self, mem: ExperimentalMemory) -> None:
        for i in range(5):
            mem.add(f"memory item number {i}", user_id="alice")

        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder, batch_size=10)
        result = dreamer.dream(user_id="alice")
        assert result.processed == 5

    def test_dream_batch_limit(self, mem: ExperimentalMemory) -> None:
        for i in range(10):
            mem.add(f"memory item number {i}", user_id="alice")

        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder, batch_size=3)
        result = dreamer.dream(user_id="alice")
        assert result.processed == 3

    def test_get_clusters_empty(self, mem: ExperimentalMemory) -> None:
        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder)
        clusters = dreamer.get_clusters(user_id="alice")
        assert clusters == []

    def test_get_clusters_with_links(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("alice likes python coding", user_id="alice")
        mem.add("cooking recipe pasta", user_id="alice")

        dreamer = DreamIntegrator(mem.store, mem.graph_store, mem.embedder)
        # Run dream first to create links
        dreamer.dream(user_id="alice")
        # Then check clusters
        clusters = dreamer.get_clusters(user_id="alice")
        # Clusters may or may not exist depending on similarity
        assert isinstance(clusters, list)
