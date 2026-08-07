#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""attributed_to / actor_id 多说话人归因单元测试 (对齐 mem0)。

覆盖:
- Memory.add(attributed_to=...) 存储 attributed_to 到 metadata
- Memory.add([{"role","content","name"}]) 从 message["name"] 提取 actor_id
- 默认无 attributed_to / actor_id
- attributed_to + expiration_date 共存
- search 返回的 metadata 透传 attributed_to / actor_id
- get_all 返回的 metadata 透传 attributed_to
"""

from __future__ import annotations

import pytest

from septmuse import Memory, MemoryConfig
from septmuse.embedders.hash import HashEmbedder


@pytest.fixture()
def mem() -> Memory:
    return Memory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class TestAddAttributedTo:
    def test_add_attributed_to_user(self, mem: Memory) -> None:
        r = mem.add("hello", user_id="alice", attributed_to="user", infer=False)
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["metadata"]["attributed_to"] == "user"

    def test_add_actor_id_from_name(self, mem: Memory) -> None:
        r = mem.add(
            [{"role": "user", "content": "hello", "name": "Maria"}],
            user_id="alice",
            infer=False,
        )
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["metadata"]["actor_id"] == "Maria"

    def test_add_without_attributed_to(self, mem: Memory) -> None:
        r = mem.add("hello", user_id="alice", infer=False)
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert "attributed_to" not in got.get("metadata", {})
        assert "actor_id" not in got.get("metadata", {})

    def test_add_attributed_to_with_expiration(self, mem: Memory) -> None:
        r = mem.add(
            "hello",
            user_id="alice",
            attributed_to="assistant",
            expiration_date="2099-12-31",
            infer=False,
        )
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["metadata"]["attributed_to"] == "assistant"
        assert got["metadata"]["expiration_date"] == "2099-12-31"

    def test_add_attributed_to_and_actor_id_both(self, mem: Memory) -> None:
        r = mem.add(
            [{"role": "user", "content": "hello", "name": "Carlos"}],
            user_id="alice",
            attributed_to="user",
            infer=False,
        )
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["metadata"]["attributed_to"] == "user"
        assert got["metadata"]["actor_id"] == "Carlos"


class TestSearchAttributedTo:
    def test_search_returns_attributed_to_in_metadata(self, mem: Memory) -> None:
        mem.add("unique alpha hello world", user_id="alice", attributed_to="user", infer=False)
        hits = mem.search("alpha hello", user_id="alice", top_k=5, threshold=0.0, hybrid=True)
        assert len(hits) == 1
        assert hits[0]["metadata"]["attributed_to"] == "user"

    def test_search_returns_actor_id_in_metadata(self, mem: Memory) -> None:
        mem.add(
            [{"role": "user", "content": "unique delta searchable text", "name": "Bob"}],
            user_id="alice",
            infer=False,
        )
        hits = mem.search("delta searchable", user_id="alice", top_k=5, threshold=0.0, hybrid=True)
        assert len(hits) == 1
        assert hits[0]["metadata"]["actor_id"] == "Bob"

    def test_search_pure_vector_returns_attributed_to(self, mem: Memory) -> None:
        mem.add("unique beta content match", user_id="alice", attributed_to="assistant", infer=False)
        hits = mem.search("beta content", user_id="alice", top_k=5, threshold=0.0, hybrid=False)
        assert len(hits) == 1
        assert hits[0]["metadata"]["attributed_to"] == "assistant"


class TestGetAllAttributedTo:
    def test_get_all_returns_attributed_to(self, mem: Memory) -> None:
        mem.add("hello world", user_id="alice", attributed_to="assistant", infer=False)
        result = mem.get_all(user_id="alice")
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["attributed_to"] == "assistant"

    def test_get_all_returns_actor_id(self, mem: Memory) -> None:
        mem.add(
            [{"role": "user", "content": "hi there", "name": "Dana"}],
            user_id="alice",
            infer=False,
        )
        result = mem.get_all(user_id="alice")
        assert len(result["results"]) == 1
        assert result["results"][0]["metadata"]["actor_id"] == "Dana"
