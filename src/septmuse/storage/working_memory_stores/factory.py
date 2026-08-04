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
"""工作记忆后端工厂 — 按配置创建 WorkingMemoryStore。

环境变量:
- SEPTMUSE_WORKING_MEMORY_BACKEND: sqlite (默认) / redis
- SEPTMUSE_REDIS_URL: redis 后端时必填
"""

from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.core.logging import get_logger
from septmuse.storage.working_memory_stores.base import WorkingMemoryStore

logger = get_logger(__name__)


def create_working_memory_store(
    config: MemoryConfig,
    *,
    engine=None,
) -> WorkingMemoryStore:
    """按配置创建工作记忆后端。

    Args:
        config: MemoryConfig (读 working_memory_backend)
        engine: 共享 SQLAlchemy engine (SQLite 路径用, 零配置)

    Returns:
        WorkingMemoryStore 实例
    """
    backend = getattr(config, "working_memory_backend", None) or "sqlite"

    if backend == "sqlite":
        if engine is None:
            raise ValueError("SQLite working memory store requires a shared engine")
        from septmuse.storage.working_memory_stores.sqlite_store import SQLiteWorkingMemoryStore

        logger.info("working_memory_store_created", backend="sqlite")
        return SQLiteWorkingMemoryStore(engine=engine)

    if backend == "redis":
        redis_url = getattr(config, "redis_url", None)
        if not redis_url:
            raise ValueError("Redis working memory store requires SEPTMUSE_REDIS_URL")
        try:
            from septmuse.storage.working_memory_stores.redis_store import RedisWorkingMemoryStore

            logger.info("working_memory_store_created", backend="redis", url=redis_url)
            return RedisWorkingMemoryStore(url=redis_url)
        except ImportError:
            logger.warning("redis_not_available", hint="pip install redis; falling back to sqlite")
            if engine is None:
                raise ValueError("Redis unavailable and no engine for SQLite fallback") from None
            from septmuse.storage.working_memory_stores.sqlite_store import SQLiteWorkingMemoryStore

            return SQLiteWorkingMemoryStore(engine=engine)

    raise ValueError(f"Unknown working memory backend: {backend}")
