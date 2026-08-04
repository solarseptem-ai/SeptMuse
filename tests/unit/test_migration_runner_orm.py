# tests/unit/test_migration_runner_orm.py
"""MigrationRunner(engine) + SQLAlchemy inspect 路径测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine

from septmuse.storage.migrations.runner import MigrationRunner


def test_migration_runner_engine_sqlite(tmp_path):
    """MigrationRunner(engine) 在 SQLite 上正确检测列。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    runner = MigrationRunner(engine)
    assert runner.backend == "sqlite"

    # 先建一张表
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE TABLE test_table (id TEXT, name TEXT)")
        conn.commit()

    # 检测列存在
    assert runner._has_column("test_table", "id") is True
    assert runner._has_column("test_table", "name") is True
    assert runner._has_column("test_table", "nonexistent") is False


def test_migration_runner_engine_has_table(tmp_path):
    """MigrationRunner(engine) 检测表存在。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig2.db'}")
    runner = MigrationRunner(engine)

    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE TABLE foo (id TEXT)")
        conn.commit()

    assert runner._has_table("foo") is True
    assert runner._has_table("nonexistent_table") is False


def test_migration_runner_engine_runs_migrations(tmp_path):
    """MigrationRunner(engine) 完整执行迁移。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig3.db'}")
    runner = MigrationRunner(engine)
    newly = runner.run()
    # 首次运行应有迁移被应用
    assert isinstance(newly, list)
    assert len(newly) > 0

    # 二次运行不应重复应用
    runner2 = MigrationRunner(engine)
    newly2 = runner2.run()
    assert len(newly2) == 0
