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
"""create_keyword_index 工厂测试 — 方言分发逻辑。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.keyword_stores.factory import create_keyword_index
from septmuse.storage.keyword_stores.mysql_fulltext import MySQLFulltextIndex
from septmuse.storage.keyword_stores.postgres_fts import PostgresFTSIndex
from septmuse.storage.keyword_stores.sqlalchemy_keyword import SQLAlchemyKeywordIndex


def test_factory_sqlite_returns_sqlalchemy_keyword():
    """SQLite 方言返回 SQLAlchemyKeywordIndex (共享 engine)。"""
    engine = create_engine("sqlite://")
    idx = create_keyword_index(engine, "sqlite")
    assert isinstance(idx, SQLAlchemyKeywordIndex)
    idx.close()


def test_factory_mysql_returns_mysql_fulltext():
    """MySQL 方言返回 MySQLFulltextIndex (SQLite engine 时内部降级)。"""
    engine = create_engine("sqlite://")
    idx = create_keyword_index(engine, "mysql")
    assert isinstance(idx, MySQLFulltextIndex)
    idx.close()


def test_factory_postgresql_returns_postgres_fts():
    """PostgreSQL 方言返回 PostgresFTSIndex (SQLite engine 时内部降级)。"""
    engine = create_engine("sqlite://")
    idx = create_keyword_index(engine, "postgresql")
    assert isinstance(idx, PostgresFTSIndex)
    idx.close()


def test_factory_unknown_dialect_raises():
    """未知方言报错。"""
    engine = create_engine("sqlite://")
    with pytest.raises(ValueError, match="Unsupported dialect"):
        create_keyword_index(engine, "oracle")


def test_factory_mysql_fallback_works():
    """MySQL 方言降级后功能正常 (SQLite engine 模拟)。"""
    engine = create_engine("sqlite://")
    idx = create_keyword_index(engine, "mysql")
    idx.add_docs({"m1": "hello world"})
    results = idx.retrieve("hello", limit=5)
    assert "m1" in results
    idx.close()


def test_factory_postgresql_fallback_works():
    """PostgreSQL 方言降级后功能正常 (SQLite engine 模拟)。"""
    engine = create_engine("sqlite://")
    idx = create_keyword_index(engine, "postgresql")
    idx.add_docs({"m1": "hello world"})
    results = idx.retrieve("hello", limit=5)
    assert "m1" in results
    idx.close()
