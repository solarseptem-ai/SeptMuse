"""迁移机制测试 — MigrationContext + MigrationRunner + 迁移模块。"""
import sqlite3

import pytest

from septmuse.storage.migrations.context import MigrationContext
from septmuse.storage.migrations.runner import MigrationRunner


@pytest.fixture
def conn():
    """临时 SQLite 连接。"""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_migration_context_has_column_false(conn):
    """has_column 在列不存在时返回 False。"""
    conn.execute("CREATE TABLE test (id TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_column("test", "name") is False


def test_migration_context_has_column_true(conn):
    """has_column 在列存在时返回 True。"""
    conn.execute("CREATE TABLE test (id TEXT, name TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_column("test", "name") is True


def test_migration_context_has_table_false(conn):
    """has_table 在表不存在时返回 False。"""
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("nonexistent") is False


def test_migration_context_has_table_true(conn):
    """has_table 在表存在时返回 True。"""
    conn.execute("CREATE TABLE test (id TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("test") is True


def test_migration_context_execute(conn):
    """execute 执行 DDL。"""
    ctx = MigrationContext(conn, "sqlite")
    ctx.execute("CREATE TABLE test (id TEXT)")
    assert ctx.has_table("test") is True


# ── MigrationRunner 测试 ──


def test_runner_empty_db_full_migration(conn):
    """空 DB → 全量迁移（5 个版本）。"""
    runner = MigrationRunner(conn, "sqlite")
    applied = runner.run()
    assert len(applied) == 5
    assert applied == ["001", "002", "003", "004", "005"]


def test_runner_skips_already_applied(conn):
    """已迁移的版本跳过。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    applied = runner.run()
    assert applied == []


def test_runner_schema_version_table_created(conn):
    """schema_version 表自动创建。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("schema_version") is True


def test_runner_records_applied_versions(conn):
    """schema_version 表记录已应用版本。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    cur = conn.execute("SELECT version FROM schema_version ORDER BY version")
    versions = [r[0] for r in cur.fetchall()]
    assert versions == ["001", "002", "003", "004", "005"]


def test_runner_idempotent(conn):
    """幂等 — 多次运行不报错，不重复。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    runner.run()
    runner.run()
    cur = conn.execute("SELECT COUNT(*) FROM schema_version")
    assert cur.fetchone()[0] == 5


def test_runner_partial_migration(conn):
    """部分迁移 — 只运行未应用的。"""
    # 手动创建 schema_version 表 + 记录 001-003
    conn.execute("CREATE TABLE schema_version (version TEXT PRIMARY KEY, description TEXT, applied_at TEXT)")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('001', 'test', '2024')")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('002', 'test', '2024')")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('003', 'test', '2024')")
    conn.commit()
    # 创建 memories 表（m001 的 DDL）
    conn.execute("CREATE TABLE memories (id TEXT, user_id TEXT, content TEXT, embedding TEXT, metadata TEXT, created_at TEXT, updated_at TEXT, is_deleted INTEGER)")
    conn.execute("CREATE TABLE history (id TEXT)")
    conn.commit()

    runner = MigrationRunner(conn, "sqlite")
    applied = runner.run()
    assert applied == ["004", "005"]

    # 检查 004 的列已添加
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "valid_at" in cols
    assert "invalid_at" in cols
    assert "expired_at" in cols

    # 检查 005 的表已创建
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "memory_access_logs" in tables
