"""m004: 添加 valid_at/invalid_at/expired_at 时态列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "004"
DESCRIPTION = "add temporal columns (valid_at/invalid_at/expired_at)"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加双时态列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS invalid_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS expired_at TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN valid_at TEXT",
            check_column=("memories", "valid_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN invalid_at TEXT",
            check_column=("memories", "invalid_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN expired_at TEXT",
            check_column=("memories", "expired_at"),
        ),
    ]
