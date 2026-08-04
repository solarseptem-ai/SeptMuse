"""迁移执行器 — 检查 schema_version 表，执行未应用的迁移。

sync 版，支持 SQLite + PG + MySQL via SQLAlchemy inspect。
通过 __init__(engine) 构造，用 engine.begin() 执行 DDL。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, text

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class MigrationRunner:
    """同步迁移执行器（SQLite + PG + MySQL via SQLAlchemy inspect）。"""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.backend = engine.dialect.name

    def _utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_schema_version_table(self) -> None:
        """创建 schema_version 追踪表（幂等）。"""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS schema_version "
                    "(version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)"
                )
            )

    def _get_applied_versions(self) -> set[str]:
        """获取已应用的迁移版本集合。"""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT version FROM schema_version"))
                return {r[0] for r in result.fetchall()}
        except Exception:
            return set()

    def _commit(self) -> None:
        """提交事务。"""
        return  # engine.begin() 自动提交

    def _has_column(self, table: str, column: str) -> bool:
        """检查列是否存在（SQLAlchemy inspect）。"""
        cols = [c["name"] for c in inspect(self._engine).get_columns(table)]
        return column in cols

    def _has_table(self, table: str) -> bool:
        """检查表是否存在。"""
        return inspect(self._engine).has_table(table)

    def run(self) -> list[str]:
        """检查 schema_version，执行未应用的迁移，返回新应用的版本列表。"""
        from septmuse.storage.migrations import MIGRATIONS

        self._ensure_schema_version_table()
        self._commit()
        applied = self._get_applied_versions()
        newly: list[str] = []
        for m in MIGRATIONS:
            if m.version in applied:
                continue
            for step in m.steps(self.backend):
                if step.check_column and self._has_column(*step.check_column):
                    continue
                if step.check_table and self._has_table(step.check_table):
                    continue
                with self._engine.begin() as conn:
                    conn.execute(text(step.sql))
            with self._engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO schema_version (version, description, applied_at) VALUES (:v, :d, :t)"),
                    {"v": m.version, "d": m.description, "t": self._utcnow_iso()},
                )
            newly.append(m.version)
            logger.info("migration_applied", version=m.version, description=m.description)
        self._commit()
        return newly
