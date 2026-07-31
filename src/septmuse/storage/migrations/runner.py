"""迁移执行器 — 检查 schema_version 表，执行未应用的迁移。

sync 版，支持 SQLite + PG。async store 在 to_thread 中调用。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class MigrationRunner:
    """同步迁移执行器（SQLite + PG）。"""

    def __init__(self, conn: Any, backend: str = "sqlite") -> None:
        self.conn = conn
        self.backend = backend

    def _utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_schema_version_table(self) -> None:
        """创建 schema_version 追踪表（幂等）。"""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     TEXT PRIMARY KEY,
                description TEXT,
                applied_at  TEXT NOT NULL
            )
            """
        )

    def _get_applied_versions(self) -> set[str]:
        """获取已应用的迁移版本集合。"""
        try:
            cur = self.conn.execute("SELECT version FROM schema_version")
            return {r[0] for r in cur.fetchall()}
        except Exception:
            return set()

    def _commit(self) -> None:
        """提交事务（PG cursor 无 commit 方法时跳过）。"""
        if hasattr(self.conn, "commit"):
            self.conn.commit()

    def _has_column(self, table: str, column: str) -> bool:
        from septmuse.storage.migrations.context import MigrationContext
        ctx = MigrationContext(self.conn, self.backend)
        return ctx.has_column(table, column)

    def _has_table(self, table: str) -> bool:
        from septmuse.storage.migrations.context import MigrationContext
        ctx = MigrationContext(self.conn, self.backend)
        return ctx.has_table(table)

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
                self.conn.execute(step.sql)
            self.conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
                (m.version, m.description, self._utcnow_iso()),
            )
            newly.append(m.version)
            logger.info("migration_applied", version=m.version, description=m.description)
        self._commit()
        return newly
