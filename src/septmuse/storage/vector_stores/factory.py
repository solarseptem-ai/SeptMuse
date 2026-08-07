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
"""create_vector_store — 方言工厂, 根据 dialect 创建 VectorStoreBase。

SQLite → SQLAlchemyVectorStore (JSON + numpy 余弦, 跨方言兼容)
PostgreSQL → PgvectorVectorStore (pgvector 扩展, 降级回退)
MySQL → SQLAlchemyVectorStore (JSON + numpy 余弦)
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from septmuse.storage.vector_stores.base import VectorStoreBase


def create_vector_store(engine: Engine, dialect: str) -> VectorStoreBase:
    """根据 dialect 创建对应的 VectorStoreBase 实现。

    Args:
        engine: SQLAlchemy Engine
        dialect: 数据库方言名 (sqlite/postgresql/mysql)

    Returns:
        VectorStoreBase 实现

    Raises:
        ValueError: 不支持的方言
    """
    if dialect == "sqlite":
        from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

        return SQLAlchemyVectorStore(engine)

    if dialect == "postgresql":
        from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore

        return PgvectorVectorStore(engine)

    if dialect == "mysql":
        from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

        return SQLAlchemyVectorStore(engine)

    raise ValueError(f"Unsupported dialect: {dialect}")
