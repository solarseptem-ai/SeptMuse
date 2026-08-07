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

向量后端: 默认 Qdrant (HNSW + BM25 稀疏向量), 也可选 chroma/sqlite/pgvector/mysql。
qdrant-client 不可用时自动降级到 SQLAlchemyVectorStore (JSON + numpy 全扫描) + 日志警告。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.configs.base import MemoryConfig
    from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
    from septmuse.storage.vector_stores.base import VectorStoreBase

logger = get_logger(__name__)


class RelationalStoreFactory:
    """根据 config 创建 ORMMemoryStore + 方言工厂的 vector_store + keyword_index。"""

    @staticmethod
    def _resolve_vector_store(config: MemoryConfig, engine: Any, dialect: str) -> VectorStoreBase:
        """解析 vector_store — chroma 不可用时降级到 SQLAlchemy (JSON + numpy)。

        chroma 后端自带 HNSW ANN 索引, 但 chromadb 是 optional extra。
        未安装时自动降级到 SQLAlchemyVectorStore (全扫描) + 日志警告, 保证零配置不 crash。
        """
        vs_backend = config.vector_store.backend
        if vs_backend == "qdrant":
            try:
                from septmuse.storage.vector_stores.qdrant import QdrantVectorStore

                return QdrantVectorStore(
                    collection_name="septmuse",
                    embedding_model_dims=config.vector_store.embedding_model_dims or 512,
                    path=os.getenv("SEPTMUSE_QDRANT_PATH"),
                    host=os.getenv("SEPTMUSE_QDRANT_HOST"),
                    port=int(os.getenv("SEPTMUSE_QDRANT_PORT", "6333"))
                    if os.getenv("SEPTMUSE_QDRANT_HOST")
                    else None,
                    url=os.getenv("SEPTMUSE_QDRANT_URL"),
                    api_key=os.getenv("SEPTMUSE_QDRANT_API_KEY"),
                    enable_bm25=True,
                )
            except ImportError:
                logger.warning("qdrant_not_available_fallback_sqlite")

        if vs_backend == "chroma":
            try:
                import chromadb  # noqa: F401

                from septmuse.storage.vector_stores.chroma import ChromaVectorStore

                persist_path = os.getenv(
                    "SEPTMUSE_CHROMA_PERSIST_PATH", str(Path.home() / ".septmuse" / "chroma")
                )
                return ChromaVectorStore(persist_path=persist_path)
            except ImportError:
                logger.warning(
                    "chroma_not_available_fallback_sqlite",
                    reason="chromadb not installed, run: pip install septmuse[chroma]",
                )
        from septmuse.storage.vector_stores.factory import create_vector_store

        return create_vector_store(engine, dialect)

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

        db_svc = DatabaseService(config)
        engine = db_svc.get_engine()
        dialect = db_svc.get_dialect()

        vector_store = RelationalStoreFactory._resolve_vector_store(config, engine, dialect)
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

        db_svc = DatabaseService(config)
        async_engine = db_svc.get_async_engine()
        dialect = db_svc.get_dialect()
        sync_engine = db_svc.get_engine()

        vector_store = RelationalStoreFactory._resolve_vector_store(config, sync_engine, dialect)
        keyword_index = create_keyword_index(sync_engine, dialect)
        return AsyncORMMemoryStore(async_engine, vector_store, keyword_index)
