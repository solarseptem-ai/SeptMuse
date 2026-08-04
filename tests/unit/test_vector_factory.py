"""create_vector_store 工厂测试 — 方言分发逻辑。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.factory import create_vector_store
from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore
from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore
from septmuse.storage.vector_stores.sqlite_vec import SQLiteVectorStore


def test_factory_sqlite_returns_sqlite_store():
    """SQLite 方言返回 SQLiteVectorStore。"""
    engine = create_engine("sqlite://")
    store = create_vector_store(engine, "sqlite")
    assert isinstance(store, SQLiteVectorStore)
    store.close()


def test_factory_mysql_returns_sqlalchemy_store():
    """MySQL 方言返回 SQLAlchemyVectorStore。"""
    engine = create_engine("sqlite://")  # 用 SQLite engine 模拟
    store = create_vector_store(engine, "mysql")
    assert isinstance(store, SQLAlchemyVectorStore)
    store.close()


def test_factory_postgresql_returns_pgvector_store():
    """PostgreSQL 方言返回 PgvectorVectorStore（降级到 SQLAlchemyVectorStore 内部）。"""
    engine = create_engine("sqlite://")  # 用 SQLite 模拟, pgvector 不可用
    store = create_vector_store(engine, "postgresql")
    assert isinstance(store, PgvectorVectorStore)
    store.close()


def test_factory_unknown_dialect_raises():
    """未知方言报错。"""
    engine = create_engine("sqlite://")
    with pytest.raises(ValueError, match="Unsupported dialect"):
        create_vector_store(engine, "oracle")
