"""m006: 添加 archived_at 列 (MemoryTable 有但旧 DB 缺)。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "006"
DESCRIPTION = "add archived_at column"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加 archived_at 列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS archived_at TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN archived_at TEXT",
            check_column=("memories", "archived_at"),
        ),
    ]
