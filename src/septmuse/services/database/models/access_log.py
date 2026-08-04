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
"""AccessLogTable — memory_access_logs 表定义（访问审计日志）。"""

from __future__ import annotations

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class AccessLogTable(SQLModel, table=True):
    """memory_access_logs 表 — 访问审计日志。"""

    __tablename__ = "memory_access_logs"

    id: str = Field(primary_key=True)
    memory_id: str = Field(index=True)
    app_id: str | None = None
    access_type: str
    metadata_json: str | None = Field(default=None, sa_column=Column("metadata", Text))
    accessed_at: str
