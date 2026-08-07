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
"""expiration_date 过期机制单元测试 (对齐 mem0 expiration_date)。

覆盖:
- _normalize_expiration_date 归一化
- _is_expired 过期判定
- Memory.add 存储 expiration_date 到 metadata
- Memory.search 默认隐藏过期记忆, show_expired=True 显示
- Memory.get_all 默认隐藏过期记忆, show_expired=True 显示
"""

from __future__ import annotations

import pytest

from septmuse import Memory, MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.main import _is_expired, _normalize_expiration_date


@pytest.fixture()
def mem() -> Memory:
    return Memory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class TestNormalizeExpirationDate:
    def test_valid_date(self) -> None:
        assert _normalize_expiration_date("2025-12-31") == "2025-12-31"

    def test_none_returns_none(self) -> None:
        assert _normalize_expiration_date(None) is None

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _normalize_expiration_date("invalid")

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            _normalize_expiration_date(12345)


class TestIsExpired:
    def test_expired_date(self) -> None:
        assert _is_expired({"expiration_date": "2020-01-01"}) is True

    def test_future_date_not_expired(self) -> None:
        assert _is_expired({"expiration_date": "2099-12-31"}) is False

    def test_empty_dict_not_expired(self) -> None:
        assert _is_expired({}) is False

    def test_none_not_expired(self) -> None:
        assert _is_expired(None) is False

    def test_invalid_date_not_expired(self) -> None:
        assert _is_expired({"expiration_date": "not-a-date"}) is False

    def test_empty_string_not_expired(self) -> None:
        assert _is_expired({"expiration_date": ""}) is False


class TestAddExpiration:
    def test_add_stores_expiration_in_metadata(self, mem: Memory) -> None:
        r = mem.add("remember this", user_id="alice", expiration_date="2099-12-31", infer=False)
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["metadata"]["expiration_date"] == "2099-12-31"

    def test_add_without_expiration_has_no_key(self, mem: Memory) -> None:
        r = mem.add("plain memory", user_id="alice", infer=False)
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert "expiration_date" not in got.get("metadata", {})


class TestSearchExpiration:
    def test_search_hides_expired_by_default(self, mem: Memory) -> None:
        text = "unique alpha content searchable"
        mem.add(text, user_id="bob", expiration_date="2020-01-01", infer=False)
        # show_expired=True 能召回
        hits_all = mem.search(text, user_id="bob", top_k=5, threshold=0.0, hybrid=False, show_expired=True)
        assert len(hits_all) == 1
        # 默认 (show_expired=False) 不返回过期记忆
        hits = mem.search(text, user_id="bob", top_k=5, threshold=0.0, hybrid=False)
        assert len(hits) == 0

    def test_search_returns_non_expired(self, mem: Memory) -> None:
        text = "live beta content searchable"
        mem.add(text, user_id="carol", expiration_date="2099-12-31", infer=False)
        hits = mem.search(text, user_id="carol", top_k=5, threshold=0.0, hybrid=False)
        assert len(hits) == 1

    def test_search_mixed_only_returns_live(self, mem: Memory) -> None:
        mem.add("expired gamma note", user_id="dave", expiration_date="2020-01-01", infer=False)
        mem.add("live gamma note", user_id="dave", expiration_date="2099-12-31", infer=False)
        # 全部召回 (含过期)
        hits_all = mem.search("gamma", user_id="dave", top_k=10, threshold=0.0, hybrid=True, show_expired=True)
        # 默认只返回未过期, 且结果中不含过期项
        hits = mem.search("gamma", user_id="dave", top_k=10, threshold=0.0, hybrid=True)
        assert len(hits) <= len(hits_all)
        for h in hits:
            assert not _is_expired(h.get("metadata"))


class TestGetAllExpiration:
    def test_get_all_hides_expired_by_default(self, mem: Memory) -> None:
        mem.add("old data", user_id="eve", expiration_date="2020-01-01", infer=False)
        mem.add("new data", user_id="eve", expiration_date="2099-12-31", infer=False)
        result = mem.get_all(user_id="eve")
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "new data"

    def test_get_all_show_expired_returns_all(self, mem: Memory) -> None:
        mem.add("old data", user_id="frank", expiration_date="2020-01-01", infer=False)
        mem.add("new data", user_id="frank", expiration_date="2099-12-31", infer=False)
        result = mem.get_all(user_id="frank", show_expired=True)
        assert len(result["results"]) == 2
