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
"""create_keyword_index — 方言工厂, 根据 dialect 创建 KeywordIndexBase。

SQLite → SQLAlchemyKeywordIndex (通用, 共享 engine, 与 memories 表同库)
PostgreSQL → PostgresFTSIndex (tsvector + ts_rank, 降级回退)
MySQL → MySQLFulltextIndex (FULLTEXT + MATCH AGAINST, 降级回退)
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from septmuse.storage.keyword_stores.base import KeywordIndexBase


def create_keyword_index(engine: Engine, dialect: str) -> KeywordIndexBase:
    """根据 dialect 创建对应的 KeywordIndexBase 实现。

    Args:
        engine: SQLAlchemy Engine
        dialect: 数据库方言名 (sqlite/postgresql/mysql)

    Returns:
        KeywordIndexBase 实现

    Raises:
        ValueError: 不支持的方言
    """
    if dialect == "sqlite":
        from septmuse.storage.keyword_stores.sqlalchemy_keyword import SQLAlchemyKeywordIndex

        # SQLite 用通用 SQLAlchemy 索引 (共享 engine, 与 memories 表同库)
        return SQLAlchemyKeywordIndex(engine)

    if dialect == "postgresql":
        from septmuse.storage.keyword_stores.postgres_fts import PostgresFTSIndex

        return PostgresFTSIndex(engine)

    if dialect == "mysql":
        from septmuse.storage.keyword_stores.mysql_fulltext import MySQLFulltextIndex

        return MySQLFulltextIndex(engine)

    raise ValueError(f"Unsupported dialect: {dialect}")
