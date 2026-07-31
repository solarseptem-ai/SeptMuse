"""迁移上下文 — 统一 SQLite/PG 的 DDL 操作接口。

为 MigrationRunner 提供 has_column/has_table/execute 方法，
屏蔽 SQLite (PRAGMA) 和 PG (information_schema) 的差异。
"""
from __future__ import annotations

from typing import Any


class MigrationContext:
    """同步迁移上下文（SQLite + PG）。"""

    def __init__(self, conn: Any, backend: str = "sqlite") -> None:
        self.conn = conn
        self.backend = backend

    def has_column(self, table: str, column: str) -> bool:
        """检查列是否存在。"""
        if self.backend == "sqlite":
            cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
        else:
            cur = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table,),
            )
            cols = [r[0] for r in cur.fetchall()]
        return column in cols

    def has_table(self, table: str) -> bool:
        """检查表是否存在。"""
        if self.backend == "sqlite":
            cur = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            r = cur.fetchone()
            return r is not None
        else:
            cur = self.conn.execute(f"SELECT to_regclass('public.{table}')")
            r = cur.fetchone()
            return r is not None and r[0] is not None

    def execute(self, sql: str) -> None:
        """执行 DDL 语句。"""
        self.conn.execute(sql)
