"""轻量级迁移注册表 — 有序迁移模块列表。

每个迁移模块导出 VERSION, DESCRIPTION, steps(backend) -> list[MigrationStep]。
MigrationRunner 按 MIGRATIONS 列表顺序执行未应用的迁移。
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass


@dataclass
class MigrationStep:
    """单个迁移步骤。

    check_column: (table, column) — 仅在列不存在时执行（SQLite ALTER TABLE 幂等性）。
    check_table: table name — 仅在表不存在时执行。
    两者都 None：无条件执行。
    """
    sql: str
    check_column: tuple[str, str] | None = None
    check_table: str | None = None


Migration = namedtuple("Migration", ["version", "description", "steps"])

from septmuse.storage.migrations import (  # noqa: E402 — MigrationStep 须先定义，子模块导入时依赖它
    m001_initial_schema,
    m002_state_columns,
    m003_session_id,
    m004_temporal,
    m005_access_logs,
    m006_archived_at,
)

_MODULES = [m001_initial_schema, m002_state_columns, m003_session_id, m004_temporal, m005_access_logs, m006_archived_at]

MIGRATIONS: list[Migration] = [
    Migration(version=m.VERSION, description=m.DESCRIPTION, steps=m.steps)
    for m in _MODULES
]
