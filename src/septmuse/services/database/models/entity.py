#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""EntityTable + EntityRelationTable — 实体 + 实体关系表定义。"""

from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class EntityTable(SQLModel, table=True):
    """septmuse_entities 表 — 实体存储（借鉴 mem0 V3 去图化）。"""

    __tablename__ = "septmuse_entities"

    id: str = Field(primary_key=True)
    entity_text: str
    entity_type: str
    entity_embedding: bytes | None = None  # BLOB 序列化向量
    linked_memory_ids: str  # JSON list[str]
    user_id: str = Field(index=True)
    agent_id: str | None = None
    created_at: str
    updated_at: str
    is_deleted: int = Field(default=0)


class EntityRelationTable(SQLModel, table=True):
    """entity_relations 表 — 实体间关系边（借鉴 graphiti）。"""

    __tablename__ = "entity_relations"
    __table_args__ = (
        UniqueConstraint("source_entity", "relation", "target_entity", "user_id"),
    )

    id: str = Field(primary_key=True)
    source_entity: str = Field(index=True)
    relation: str
    target_entity: str = Field(index=True)
    user_id: str = Field(index=True)
    memory_id: str | None = None
    created_at: str | None = None
