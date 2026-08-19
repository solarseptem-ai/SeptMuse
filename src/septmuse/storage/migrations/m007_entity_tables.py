"""m007: 实体表 + 实体关系表纳入迁移追踪。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "007"
DESCRIPTION = "entity tables (septmuse_entities + entity_relations)"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """创建 septmuse_entities + entity_relations 表 (如不存在)。"""
    return [
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS septmuse_entities (
                id                 TEXT PRIMARY KEY,
                entity_text        TEXT NOT NULL,
                entity_type        TEXT,
                entity_embedding   BLOB,
                linked_memory_ids  TEXT,
                user_id            TEXT NOT NULL,
                agent_id           TEXT,
                created_at         TEXT,
                updated_at         TEXT,
                is_deleted         INTEGER DEFAULT 0
            )
            """,
            check_table="septmuse_entities",
        ),
        MigrationStep(
            "CREATE INDEX IF NOT EXISTS idx_se_user ON septmuse_entities(user_id)",
        ),
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS entity_relations (
                id            TEXT PRIMARY KEY,
                source_entity TEXT NOT NULL,
                relation      TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                user_id       TEXT NOT NULL,
                memory_id     TEXT,
                created_at    TEXT,
                UNIQUE(source_entity, relation, target_entity, user_id)
            )
            """,
            check_table="entity_relations",
        ),
        MigrationStep(
            "CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_entity, user_id)",
        ),
        MigrationStep(
            "CREATE INDEX IF NOT EXISTS idx_er_target ON entity_relations(target_entity, user_id)",
        ),
    ]
