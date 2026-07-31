"""m005: 创建 memory_access_logs 审计日志表。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "005"
DESCRIPTION = "create memory_access_logs table"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """创建访问日志表 + 索引。"""
    return [
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS memory_access_logs (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT NOT NULL,
                app_id       TEXT,
                access_type  TEXT NOT NULL,
                metadata     TEXT,
                accessed_at  TEXT NOT NULL
            )
            """
        ),
        MigrationStep(
            "CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id)"
        ),
    ]
