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
"""数据库表定义包 — 跨方言 DDL（SQLite / MySQL / PostgreSQL）。

SQLModel.metadata.create_all(engine) 会根据 engine dialect 自动生成对应方言的 DDL。
"""

from septmuse.services.database.models.access_log import AccessLogTable
from septmuse.services.database.models.entity import EntityRelationTable, EntityTable
from septmuse.services.database.models.history import HistoryTable
from septmuse.services.database.models.memory import MemoryTable

__all__ = [
    "AccessLogTable",
    "EntityRelationTable",
    "EntityTable",
    "HistoryTable",
    "MemoryTable",
]
