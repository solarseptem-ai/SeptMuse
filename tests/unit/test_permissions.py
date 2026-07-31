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
"""check_memory_access_permissions 4 层权限检查测试。"""

from __future__ import annotations

import pytest

from septmuse.governance.permissions import MemoryState, check_memory_access_permissions
from septmuse.storage.sqlite.store import SQLiteMemoryStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    # Task 1 测试需要 state 列, 但 Task 3 才正式加迁移函数
    # 这里手动 ALTER TABLE (Task 3 的 _migrate_add_state_columns 会处理重复)
    cols = {row[1] for row in s.conn.execute("PRAGMA table_info(memories)")}
    if "state" not in cols:
        s.conn.execute("ALTER TABLE memories ADD COLUMN state TEXT DEFAULT 'active'")
        s.conn.commit()
    yield s
    s.close()


def _add_and_set_state(store, memory_id, content, state=None):
    """辅助: add 记忆, 可选手动设 state。"""
    mid = store.add(content, [1.0, 0.0], user_id="alice") if memory_id is None else memory_id
    if state is not None:
        store.conn.execute("UPDATE memories SET state = ? WHERE id = ?", (state, mid))
        store.conn.commit()
    return mid


def test_check_nonexistent_memory_returns_false(store):
    allowed, reason = check_memory_access_permissions(store, "nonexistent", None)
    assert allowed is False
    assert "not found" in reason


def test_check_deleted_state_returns_false(store):
    mid = _add_and_set_state(store, None, "to delete", "deleted")
    allowed, reason = check_memory_access_permissions(store, mid, None)
    assert allowed is False
    assert "deleted" in reason


def test_check_paused_state_returns_false(store):
    mid = _add_and_set_state(store, None, "paused mem", "paused")
    allowed, reason = check_memory_access_permissions(store, mid, None)
    assert allowed is False
    assert "paused" in reason


def test_check_archived_state_returns_false(store):
    mid = _add_and_set_state(store, None, "archived mem", "archived")
    allowed, reason = check_memory_access_permissions(store, mid, None)
    assert allowed is False
    assert "archived" in reason


def test_check_active_no_app_id_returns_true(store):
    mid = store.add("active mem", [1.0, 0.0], user_id="alice")
    allowed, reason = check_memory_access_permissions(store, mid, None)
    assert allowed is True
    assert "self access" in reason


def test_check_active_with_app_id_returns_true(store):
    mid = store.add("active mem", [1.0, 0.0], user_id="alice")
    allowed, reason = check_memory_access_permissions(store, mid, "myapp")
    assert allowed is True
    assert "myapp" in reason


def test_check_empty_app_id_returns_false(store):
    mid = store.add("active mem", [1.0, 0.0], user_id="alice")
    allowed, reason = check_memory_access_permissions(store, mid, "")
    assert allowed is False
    assert "empty" in reason


def test_check_none_state_treated_as_active(store):
    """旧数据 state 可能是 NULL — 应视为 active。"""
    mid = store.add("old mem", [1.0, 0.0], user_id="alice")
    store.conn.execute("UPDATE memories SET state = NULL WHERE id = ?", (mid,))
    store.conn.commit()
    allowed, _reason = check_memory_access_permissions(store, mid, None)
    assert allowed is True


def test_memory_state_enum_values():
    assert MemoryState.ACTIVE == "active"
    assert MemoryState.PAUSED == "paused"
    assert MemoryState.ARCHIVED == "archived"
    assert MemoryState.DELETED == "deleted"
