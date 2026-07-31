#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""state 状态机 + ALTER TABLE 迁移 + memory_access_logs 测试。"""

from __future__ import annotations

import pytest

from septmuse.storage.sqlite.store import SQLiteMemoryStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_add_sets_state_active(store):
    mid = store.add("hello", [1.0, 0.0], user_id="alice")
    row = store.conn.execute("SELECT state FROM memories WHERE id = ?", (mid,)).fetchone()
    assert row[0] == "active"


def test_delete_sets_state_deleted(store):
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    row = store.conn.execute("SELECT state, is_deleted, deleted_at FROM memories WHERE id = ?", (mid,)).fetchone()
    assert row[0] == "deleted"
    assert row[1] == 1  # is_deleted 并存
    assert row[2] is not None  # deleted_at


def test_get_all_filters_non_active(store):
    m1 = store.add("active1", [1.0, 0.0], user_id="alice")
    m2 = store.add("active2", [0.0, 1.0], user_id="alice")
    store.delete(m2)
    results = store.get_all(user_id="alice")
    ids = [r["id"] for r in results]
    assert m1 in ids
    assert m2 not in ids


def test_search_filters_non_active(store):
    m1 = store.add("active", [1.0, 0.0], user_id="alice")
    m2 = store.add("deleted", [1.0, 0.0], user_id="alice")
    store.delete(m2)
    results = store.search([1.0, 0.0], user_id="alice", top_k=10)
    ids = [r["id"] for r in results]
    assert m1 in ids
    assert m2 not in ids


def test_record_access_log_creates_entry(store):
    mid = store.add("logged", [1.0, 0.0], user_id="alice")
    log_id = store._record_access_log(mid, "app1", "get", {"k": "v"})
    assert log_id is not None
    logs = store.get_access_logs(mid)
    assert len(logs) == 1
    assert logs[0]["access_type"] == "get"
    assert logs[0]["app_id"] == "app1"
    assert logs[0]["metadata"] == {"k": "v"}


def test_get_access_logs_ordered_desc(store):
    mid = store.add("logged", [1.0, 0.0], user_id="alice")
    store._record_access_log(mid, "app1", "get", None)
    store._record_access_log(mid, "app1", "search", None)
    store._record_access_log(mid, "app1", "delete", None)
    logs = store.get_access_logs(mid)
    assert len(logs) == 3
    # ORDER BY accessed_at DESC — 最新在前
    assert logs[0]["access_type"] == "delete"
    assert logs[2]["access_type"] == "get"


def test_get_access_logs_limit(store):
    mid = store.add("logged", [1.0, 0.0], user_id="alice")
    for _ in range(5):
        store._record_access_log(mid, "app1", "get", None)
    logs = store.get_access_logs(mid, limit=3)
    assert len(logs) == 3


def test_old_data_migration_sets_active(store):
    """模拟旧 DB: 直接 INSERT 无 state 列 → 迁移后 state='active'。"""
    # 先建一个旧 memories 表 (无 state 列)
    store.conn.execute("DROP TABLE memories")
    store.conn.execute(
        "CREATE TABLE memories (id TEXT PRIMARY KEY, user_id TEXT, content TEXT, embedding TEXT, "
        "metadata TEXT, is_deleted INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)"
    )
    store.conn.execute(
        "INSERT INTO memories (id, user_id, content, embedding, metadata, is_deleted, created_at, updated_at) "
        "VALUES ('old1', 'alice', 'old', '[1.0]', '{}', 0, '2025-01-01', '2025-01-01')"
    )
    store.conn.commit()
    # 触发迁移
    store._migrate_add_state_columns()
    # 验证 state 默认 'active'
    row = store.conn.execute("SELECT state FROM memories WHERE id = 'old1'").fetchone()
    assert row[0] == "active"


def test_columns_not_duplicated_on_re_migration(store):
    """重复迁移不应报错。"""
    store._migrate_add_state_columns()
    store._migrate_add_state_columns()  # 第二次
    # 验证只有一列 state
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert cols.count("state") == 1
    assert cols.count("app_id") == 1
