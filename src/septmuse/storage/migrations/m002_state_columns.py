"""m002: 添加 state/deleted_at/app_id 列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "002"
DESCRIPTION = "add state/deleted_at/app_id columns"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加状态机列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS state TEXT DEFAULT 'active'"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS app_id TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN state TEXT DEFAULT 'active'",
            check_column=("memories", "state"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN deleted_at TEXT",
            check_column=("memories", "deleted_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN app_id TEXT",
            check_column=("memories", "app_id"),
        ),
    ]
