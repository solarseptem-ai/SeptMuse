#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""state 状态机 + 建表幂等 + memory_access_logs 测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture()
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    s = ORMMemoryStore(engine)
    yield s
    s.close()


def test_add_sets_state_active(store):
    mid = store.add("hello", [1.0, 0.0], user_id="alice")
    with store.engine.connect() as conn:
        row = conn.execute(text("SELECT state FROM memories WHERE id = :mid"), {"mid": mid}).fetchone()
    assert row[0] == "active"


def test_delete_sets_state_deleted(store):
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    with store.engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, is_deleted, deleted_at FROM memories WHERE id = :mid"), {"mid": mid}
        ).fetchone()
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
    """ORMMemoryStore 用 SQLModel 建表, state 列 NOT NULL + default='active'。

    旧路径的 ALTER TABLE 迁移 (添加 state 列) 在 ORMMemoryStore 中不存在:
    SQLModel.metadata.create_all 一次性创建完整 schema。
    此处验证: 通过 ORM 插入的记忆 state 默认 'active'。
    """
    mid = store.add("old data", [1.0, 0.0], user_id="alice")
    with store.engine.connect() as conn:
        row = conn.execute(text("SELECT state FROM memories WHERE id = :mid"), {"mid": mid}).fetchone()
    assert row[0] == "active"


def test_columns_not_duplicated_on_re_migration(store):
    """重复建表不应报错 (SQLModel.metadata.create_all 幂等)。"""
    store._create_tables()
    store._create_tables()  # 第二次
    cols = [c["name"] for c in inspect(store.engine).get_columns("memories")]
    assert cols.count("state") == 1
    assert cols.count("app_id") == 1
