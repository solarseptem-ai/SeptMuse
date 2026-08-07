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
"""数据库配置 (借鉴 Langflow DatabaseService 配置 + mem0 history_db_path)。

SeptMuse 默认 SQLite 单文件, 可选 PostgreSQL (pgvector)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """数据库连接配置。

    零配置: db_url=None + db_path=None → 默认 ~/.septmuse/septmuse.db。
    内存库: db_path=":memory:"。
    切换 MySQL/PG: 设 db_url="mysql://user:pass@host:3306/septmuse" 或
                   db_url="postgresql://user:pass@host:5432/septmuse"。
    """

    db_url: str | None = Field(
        default=None,
        description="数据库连接 URL; None=用 db_path 回退 SQLite; mysql:// / postgresql:// 切换后端",
    )
    db_path: str | Path | None = Field(
        default=None,
        description="SQLite 路径; None → 默认 ~/.septmuse/septmuse.db; ':memory:' → 内存库",
    )
    connection_pool_size: int = Field(default=5, description="连接池大小 (PG 模式)")
    connection_max_overflow: int = Field(default=10, description="连接池溢出上限 (PG 模式)")
    connect_timeout: float = Field(default=30.0, description="连接超时秒数")
    sqlite_pragmas: dict[str, Any] = Field(
        default_factory=lambda: {"journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 5000},
        description="SQLite PRAGMA 设置 (WAL 模式提升并发读写, busy_timeout 毫秒级等待)",
    )
