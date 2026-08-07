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
"""update 实体重链接测试 — 文本变更时清理旧实体 + 链接新实体。"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine

from septmuse.configs.defaults import default_config
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.main import Memory
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


def _make_memory(tmp_path):
    """构造带 entity_store 的 Memory (ORMMemoryStore 路径 + HashEmbedder)。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'relink.db'}")
    store = ORMMemoryStore(engine)
    config = default_config()
    config.database.db_path = str(tmp_path / "relink.db")
    return Memory(config=config, store=store, embedder=HashEmbedder())


def _linked_ids(entity_dict):
    """从 search/list 返回的实体 dict 解析 linked_memory_ids (raw JSON string → list)。"""
    raw = entity_dict["linked_memory_ids"]
    return json.loads(raw) if isinstance(raw, str) else raw


class TestUpdateEntityRelink:
    def test_relink_entities_on_text_change(self, tmp_path):
        """文本从 'I like Python' → 'I like Rust': 旧 Python 实体清理 + 新 Rust 实体链接。"""
        m = _make_memory(tmp_path)
        assert m.entity_store is not None, "ORM 路径下 entity_store 不应为 None"

        # add → "Python" 实体链接到 memory_id
        result = m.add("I like Python", user_id="alice")
        mid = result["results"][0]["id"]

        python_hits = m.entity_store.search("Python", user_id="alice")
        assert len(python_hits) >= 1, "add 后应有 Python 实体"
        assert mid in _linked_ids(python_hits[0]), "Python 实体应链接到 memory_id"

        # update → 文本变更
        m.update(mid, "I like Rust", user_id="alice")

        # 旧 "Python" 实体不再链接 memory_id (单链接时被软删除)
        python_after = m.entity_store.search("Python", user_id="alice")
        for ent in python_after:
            assert mid not in _linked_ids(ent), "update 后 Python 实体不应再链接 memory_id"

        # 新 "Rust" 实体存在且链接 memory_id
        rust_hits = m.entity_store.search("Rust", user_id="alice")
        assert len(rust_hits) >= 1, "update 后应有 Rust 实体"
        assert mid in _linked_ids(rust_hits[0]), "Rust 实体应链接到 memory_id"

    def test_no_relink_when_text_unchanged(self, tmp_path):
        """只改 metadata (content 不变) 时不触发实体重链接。"""
        m = _make_memory(tmp_path)
        result = m.add("I like Python", user_id="alice")
        mid = result["results"][0]["id"]

        # 只改 metadata
        m.update(mid, metadata={"tag": "important"}, user_id="alice")

        # Python 实体仍应链接 memory_id
        python_hits = m.entity_store.search("Python", user_id="alice")
        assert len(python_hits) >= 1, "文本未变, Python 实体应仍存在"
        assert mid in _linked_ids(python_hits[0]), "文本未变, Python 实体应仍链接 memory_id"

    def test_no_relink_when_same_content(self, tmp_path):
        """传入相同 content 也不触发实体重链接 (text_changed=False)。"""
        m = _make_memory(tmp_path)
        result = m.add("I like Python", user_id="alice")
        mid = result["results"][0]["id"]

        # 传入相同 content
        m.update(mid, "I like Python", user_id="alice")

        python_hits = m.entity_store.search("Python", user_id="alice")
        assert len(python_hits) >= 1, "content 相同, Python 实体应仍存在"
        assert mid in _linked_ids(python_hits[0]), "content 相同, Python 实体应仍链接 memory_id"

    def test_update_not_found_no_crash(self, tmp_path):
        """更新不存在的 memory_id 不崩溃 (无实体可清理)。"""
        m = _make_memory(tmp_path)
        result = m.update("nonexistent-id", "new content", user_id="alice")
        assert result["event"] == "NOT_FOUND"

    def test_update_without_user_id_uses_default(self, tmp_path):
        """user_id=None 时实体重链接用 'default' (向后兼容)。"""
        m = _make_memory(tmp_path)
        result = m.add("I like Python", user_id="default")
        mid = result["results"][0]["id"]

        # update 不传 user_id (默认 None → entity relink 用 'default')
        m.update(mid, "I like Rust")

        # Rust 实体应在 'default' 用户下
        rust_hits = m.entity_store.search("Rust", user_id="default")
        assert len(rust_hits) >= 1, "user_id=None 时应在 default 用户下创建 Rust 实体"
        assert mid in _linked_ids(rust_hits[0]), "Rust 实体应链接到 memory_id"
