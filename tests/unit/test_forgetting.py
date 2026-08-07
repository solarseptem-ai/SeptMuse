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
"""阶段4 §6.2 遗忘曲线单元测试 — MemoryStrength + ForgettingRetriever。

固化 (架构文档 §6.2 自研):
- MemoryStrength: decay (R=exp(-t/S)) + rehearse (strength回升)
- ForgettingRetriever: apply_strength (final_score=relevance×strength) + rehearse + archive
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.models.strength import (
    ARCHIVE_THRESHOLD,
    REHEARSE_BASE_VALUE_THRESHOLD,
    REHEARSE_STRENGTH_THRESHOLD,
    MemoryStrength,
)
from septmuse.retrieval.forgetting import ForgettingRetriever

UTC = timezone.utc


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


# ======================================================================
# MemoryStrength
# ======================================================================


class TestMemoryStrength:
    def test_initial_strength(self) -> None:
        s = MemoryStrength(memory_id="m1", user_id="alice")
        assert s.strength == 1.0
        assert s.access_count == 0
        assert not s.archived

    def test_decay_no_time_elapsed(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=1.0)
        s.last_accessed = now
        decayed = s.decay(now)
        assert decayed == pytest.approx(1.0)

    def test_decay_decreases_over_time(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=1.0)
        s.last_accessed = now - timedelta(hours=24)
        decayed = s.decay(now)
        # After 1 day with base_value=1.0, R = exp(-1) ≈ 0.37
        assert decayed < 0.5
        assert decayed > 0.3

    def test_decay_higher_base_value_slower(self) -> None:
        now = datetime.now(UTC)
        s_low = MemoryStrength(memory_id="m1", user_id="alice", base_value=0.1)
        s_low.last_accessed = now - timedelta(hours=1)
        s_high = MemoryStrength(memory_id="m2", user_id="alice", base_value=1.0)
        s_high.last_accessed = now - timedelta(hours=1)
        assert s_high.decay(now) > s_low.decay(now)

    def test_rehearse_recovers_strength(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=1.0, strength=0.3)
        s.last_accessed = now - timedelta(hours=48)
        # Before rehearse: decayed strength is very low
        assert s.decay(now) < 0.2
        # Rehearse
        s.rehearse(now)
        assert s.strength > 0.2  # recovered
        assert s.access_count == 1
        assert s.last_accessed == now

    def test_rehearse_unarchives(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", archived=True, base_value=1.0)
        s.rehearse(now)
        assert not s.archived

    def test_should_rehearse_true(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=0.8)
        s.last_accessed = now - timedelta(hours=48)  # decayed below 0.3
        assert s.should_rehearse(now)

    def test_should_rehearse_false_low_base_value(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=0.3)  # too low
        s.last_accessed = now - timedelta(hours=48)
        assert not s.should_rehearse(now)

    def test_should_rehearse_false_recently_accessed(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=1.0)
        s.last_accessed = now  # just accessed, strength high
        assert not s.should_rehearse(now)

    def test_should_archive_true(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=0.1)
        s.last_accessed = now - timedelta(days=30)
        assert s.should_archive(now)

    def test_should_archive_false_recently_accessed(self) -> None:
        now = datetime.now(UTC)
        s = MemoryStrength(memory_id="m1", user_id="alice", base_value=1.0)
        s.last_accessed = now
        assert not s.should_archive(now)

    def test_thresholds(self) -> None:
        assert REHEARSE_STRENGTH_THRESHOLD == 0.3
        assert REHEARSE_BASE_VALUE_THRESHOLD == 0.5
        assert ARCHIVE_THRESHOLD == 0.1


# ======================================================================
# ForgettingRetriever
# ======================================================================


class TestForgettingRetriever:
    def test_apply_strength_basic(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        results = mem.search("alice", user_id="alice")
        assert len(results) > 0

        retriever = ForgettingRetriever(mem.typed_store)
        weighted = retriever.apply_strength(results, user_id="alice")
        assert len(weighted) > 0
        assert weighted[0].final_score > 0
        # 加权平均: final = 0.7*relevance + 0.3*strength, 两者都在[0,1]

    def test_apply_strength_empty(self, mem: ExperimentalMemory) -> None:
        retriever = ForgettingRetriever(mem.typed_store)
        weighted = retriever.apply_strength([], user_id="alice")
        assert weighted == []

    def test_apply_strength_sorts_by_final_score(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("bob likes java", user_id="alice")
        results = mem.search("likes", user_id="alice")
        retriever = ForgettingRetriever(mem.typed_store)
        weighted = retriever.apply_strength(results, user_id="alice")
        if len(weighted) >= 2:
            assert weighted[0].final_score >= weighted[1].final_score

    def test_find_rehearse_candidates_empty(self, mem: ExperimentalMemory) -> None:
        retriever = ForgettingRetriever(mem.typed_store)
        candidates = retriever.find_rehearse_candidates(user_id="alice")
        assert candidates == []

    def test_find_rehearse_candidates_stale(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        # Create strength record with high base_value and old last_accessed
        mid = mem.store.get_all(user_id="alice")[0]["id"]
        s = mem.typed_store.get_or_create_strength(mid, user_id="alice", base_value=0.9)
        # Set last_accessed to 3 days ago and persist
        old_time = datetime.now(UTC) - timedelta(days=3)
        decayed = s.decay(old_time)
        mem.typed_store.update_strength(
            mid, user_id="alice", strength=decayed, last_accessed=old_time.replace(tzinfo=None)
        )

        retriever = ForgettingRetriever(mem.typed_store)
        candidates = retriever.find_rehearse_candidates(user_id="alice")
        # Should find the stale high-value memory
        assert len(candidates) >= 1

    def test_rehearse_single(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mid = mem.store.get_all(user_id="alice")[0]["id"]
        retriever = ForgettingRetriever(mem.typed_store)
        result = retriever.rehearse(mid, user_id="alice")
        assert result is not None
        assert result.access_count == 1

    def test_rehearse_batch(self, mem: ExperimentalMemory) -> None:
        mem.add("memory 1", user_id="alice")
        mem.add("memory 2", user_id="alice")
        ids = [m["id"] for m in mem.store.get_all(user_id="alice")]
        retriever = ForgettingRetriever(mem.typed_store)
        count = retriever.rehearse_batch(ids, user_id="alice")
        assert count == 2

    def test_archive_stale_empty(self, mem: ExperimentalMemory) -> None:
        retriever = ForgettingRetriever(mem.typed_store)
        archived = retriever.archive_stale(user_id="alice")
        assert archived == []

    def test_archived_filtered_from_results(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mid = mem.store.get_all(user_id="alice")[0]["id"]
        # Manually archive
        mem.typed_store.get_or_create_strength(mid, user_id="alice")
        mem.typed_store.update_strength(mid, user_id="alice", strength=0.05, archived=True)

        results = mem.search("alice", user_id="alice")
        retriever = ForgettingRetriever(mem.typed_store)
        weighted = retriever.apply_strength(results, user_id="alice")
        # Archived memory should be filtered out
        assert all(w.id != mid for w in weighted)
