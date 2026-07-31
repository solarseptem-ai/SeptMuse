"""m001: 初始 schema — memories + history 表。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "001"
DESCRIPTION = "initial schema (memories + history)"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """创建 memories + history 基础表。"""
    return [
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                agent_id    TEXT,
                content     TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                is_deleted  INTEGER DEFAULT 0
            )
            """
        ),
        MigrationStep("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"),
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS history (
                id          TEXT PRIMARY KEY,
                memory_id   TEXT,
                old_memory  TEXT,
                new_memory  TEXT,
                event       TEXT,
                created_at  TEXT,
                is_deleted  INTEGER
            )
            """
        ),
    ]
