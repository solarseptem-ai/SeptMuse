"""m003: 添加 session_id 列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "003"
DESCRIPTION = "add session_id column"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加会话 ID 列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS session_id TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN session_id TEXT",
            check_column=("memories", "session_id"),
        ),
    ]
