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
"""RelationalStoreFactory — 根据 config 创建 ORMMemoryStore / AsyncORMMemoryStore。

零配置回退: 无 db_url 时由 Memory facade 走 ORMMemoryStore (不走本工厂)。
有 db_url (SEPTMUSE_DB_URL 或 config.database.db_url) 时:
  sync  → ORMMemoryStore (engine + vector_store + keyword_index)
  async → AsyncORMMemoryStore (async_engine + sync vector_store + keyword_index)

双写 vector_store/keyword_index 用 sync engine (ORMMemoryStore 双写是 sync 调用)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from septmuse.configs.base import MemoryConfig
    from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


class RelationalStoreFactory:
    """根据 config 创建 ORMMemoryStore + 方言工厂的 vector_store + keyword_index。"""

    @staticmethod
    def create(config: MemoryConfig) -> ORMMemoryStore:
        """创建 sync ORMMemoryStore。

        Args:
            config: MemoryConfig (需有 database.db_url)

        Returns:
            ORMMemoryStore 实例 (engine + vector_store + keyword_index)
        """
        from septmuse.services.database.service import DatabaseService
        from septmuse.storage.keyword_stores.factory import create_keyword_index
        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
        from septmuse.storage.vector_stores.factory import create_vector_store

        db_svc = DatabaseService(config)
        engine = db_svc.get_engine()
        dialect = db_svc.get_dialect()
        vector_store = create_vector_store(engine, dialect)
        keyword_index = create_keyword_index(engine, dialect)
        return ORMMemoryStore(engine, vector_store, keyword_index)

    @staticmethod
    def create_async(config: MemoryConfig) -> AsyncORMMemoryStore:
        """创建 async AsyncORMMemoryStore。

        Args:
            config: MemoryConfig (需有 database.db_url)

        Returns:
            AsyncORMMemoryStore 实例 (async_engine + sync vector_store + keyword_index)
        """
        from septmuse.services.database.service import DatabaseService
        from septmuse.storage.keyword_stores.factory import create_keyword_index
        from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore
        from septmuse.storage.vector_stores.factory import create_vector_store

        db_svc = DatabaseService(config)
        async_engine = db_svc.get_async_engine()
        dialect = db_svc.get_dialect()
        # 双写 vector_store/keyword_index 用 sync engine (ORMMemoryStore 双写是 sync 调用)
        sync_engine = db_svc.get_engine()
        vector_store = create_vector_store(sync_engine, dialect)
        keyword_index = create_keyword_index(sync_engine, dialect)
        return AsyncORMMemoryStore(async_engine, vector_store, keyword_index)
