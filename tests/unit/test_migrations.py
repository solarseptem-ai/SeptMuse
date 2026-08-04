"""6 个迁移模块 DDL 正确性测试。"""
import sqlite3

import pytest

from septmuse.storage.migrations import MIGRATIONS


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_migrations_list_has_6():
    """注册表有 6 个迁移。"""
    assert len(MIGRATIONS) == 6


def test_migrations_versions_sequential():
    """版本号从 001 到 006 有序。"""
    versions = [m.version for m in MIGRATIONS]
    assert versions == ["001", "002", "003", "004", "005", "006"]


def test_m001_creates_memories_and_history(conn):
    """m001 创建 memories + history 表。"""
    m001 = MIGRATIONS[0]
    assert m001.version == "001"
    for step in m001.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "memories" in tables
    assert "history" in tables


def test_m002_adds_state_columns(conn):
    """m002 添加 state/deleted_at/app_id 列。"""
    conn.execute("CREATE TABLE memories (id TEXT, user_id TEXT, content TEXT, embedding TEXT, metadata TEXT, created_at TEXT, updated_at TEXT, is_deleted INTEGER)")
    m002 = MIGRATIONS[1]
    assert m002.version == "002"
    for step in m002.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "state" in cols
    assert "deleted_at" in cols
    assert "app_id" in cols


def test_m003_adds_session_id(conn):
    """m003 添加 session_id 列。"""
    conn.execute("CREATE TABLE memories (id TEXT)")
    m003 = MIGRATIONS[2]
    assert m003.version == "003"
    for step in m003.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "session_id" in cols


def test_m004_adds_temporal_columns(conn):
    """m004 添加 valid_at/invalid_at/expired_at 列。"""
    conn.execute("CREATE TABLE memories (id TEXT)")
    m004 = MIGRATIONS[3]
    assert m004.version == "004"
    for step in m004.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "valid_at" in cols
    assert "invalid_at" in cols
    assert "expired_at" in cols


def test_m005_creates_access_logs(conn):
    """m005 创建 memory_access_logs 表。"""
    m005 = MIGRATIONS[4]
    assert m005.version == "005"
    for step in m005.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "memory_access_logs" in tables


def test_m006_adds_archived_at(conn):
    """m006 添加 archived_at 列。"""
    conn.execute("CREATE TABLE memories (id TEXT)")
    m006 = MIGRATIONS[5]
    assert m006.version == "006"
    for step in m006.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "archived_at" in cols
