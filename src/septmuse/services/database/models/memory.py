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
"""MemoryTable — memories 表定义（跨方言 DDL）。"""

from __future__ import annotations

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class MemoryTable(SQLModel, table=True):
    """memories 表 — 记忆主表。"""

    __tablename__ = "memories"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    agent_id: str | None = None
    session_id: str | None = None
    content: str
    # metadata 是 Python 保留名, 用 sa_column 映射到数据库列名
    metadata_json: str = Field(default="{}", sa_column=Column("metadata", Text))
    created_at: str | None = None
    updated_at: str | None = None
    is_deleted: int = Field(default=0)
    state: str = Field(default="active")
    app_id: str | None = None
    archived_at: str | None = None
    deleted_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None
