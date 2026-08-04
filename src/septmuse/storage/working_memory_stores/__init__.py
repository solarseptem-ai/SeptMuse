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
"""工作记忆独立后端 — WorkingMemoryStore ABC + SQLite/Redis 实现 + 工厂。"""

from septmuse.storage.working_memory_stores.base import WorkingMemoryStore
from septmuse.storage.working_memory_stores.factory import create_working_memory_store
from septmuse.storage.working_memory_stores.sqlite_store import SQLiteWorkingMemoryStore

__all__ = [
    "RedisWorkingMemoryStore",
    "SQLiteWorkingMemoryStore",
    "WorkingMemoryStore",
    "create_working_memory_store",
]


def __getattr__(name: str):  # PEP 562: 延迟 import, redis 未安装时不炸
    if name == "RedisWorkingMemoryStore":
        from septmuse.storage.working_memory_stores.redis_store import RedisWorkingMemoryStore

        return RedisWorkingMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
