#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""DatabaseService 单元测试 — 引擎管理 + 跨方言建表。"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from septmuse.configs.base import MemoryConfig
from septmuse.configs.database import DatabaseConfig
from septmuse.services.database.factory import DatabaseServiceFactory
from septmuse.services.database.service import DatabaseService


def _make_config(db_path: str) -> MemoryConfig:
    """构造指向 tmp_path 的 MemoryConfig。"""
    config = MemoryConfig()
    config.database = DatabaseConfig(db_path=db_path)
    return config


@pytest.fixture()
def db_service(tmp_path, monkeypatch):
    """每个测试用独立 tmp_path SQLite, 测试后 dispose。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    config = _make_config(str(tmp_path / "test.db"))
    svc = DatabaseService(config=config)
    yield svc
    svc.engine.dispose()


def test_sqlite_engine_created(tmp_path, monkeypatch):
    """零配置默认 SQLite。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    config = _make_config(str(tmp_path / "test.db"))
    svc = DatabaseService(config=config)
    assert svc.get_dialect() == "sqlite"
    svc.engine.dispose()


def test_create_tables_delegates_to_store(db_service):
    """建表委托给 store, DatabaseService 不建表。"""
    db_service.create_tables()  # 不报错即可
    # memories 表由 ORMMemoryStore._create_tables 建, DatabaseService 不管


def test_engine_query_works(tmp_path, monkeypatch):
    """建表后可执行 SQL（用 store 自己建表）。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    from sqlalchemy import create_engine

    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    store = ORMMemoryStore(engine)
    store.close()
    # 用 DatabaseService 的 engine 验证表存在
    config = _make_config(str(tmp_path / "test.db"))
    svc = DatabaseService(config=config)
    with svc.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM memories")).fetchone()
    assert result[0] == 0
    svc.engine.dispose()


def test_dialect_detection_sqlite(tmp_path, monkeypatch):
    """db_url=sqlite:// → dialect=sqlite。"""
    monkeypatch.setenv("SEPTMUSE_DB_URL", "sqlite:///" + str(tmp_path / "x.db"))
    svc = DatabaseService(config=None)
    assert svc.get_dialect() == "sqlite"
    svc.engine.dispose()


def test_safe_url_hides_password(tmp_path, monkeypatch):
    """日志安全 URL 隐藏密码。"""
    monkeypatch.setenv("SEPTMUSE_DB_URL", "sqlite:///" + str(tmp_path / "x.db"))
    svc = DatabaseService(config=None)
    svc.database_url = "mysql://user:secret@host:3306/septmuse"
    assert "***" in svc._safe_url()
    svc.engine.dispose()


def test_factory_create(tmp_path, monkeypatch):
    """工厂模式创建 DatabaseService。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    config = _make_config(str(tmp_path / "test.db"))
    factory = DatabaseServiceFactory()
    svc = factory.create(config=config)
    assert svc.get_dialect() == "sqlite"
    svc.engine.dispose()


def test_session_maker(tmp_path, monkeypatch):
    """session_maker 可创建 session。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    from sqlalchemy import create_engine

    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

    db_path = str(tmp_path / "test.db")
    engine = create_engine(f"sqlite:///{db_path}")
    store = ORMMemoryStore(engine)
    store.close()
    config = _make_config(db_path)
    svc = DatabaseService(config=config)
    Session = svc.get_session_maker()
    with Session() as session:
        result = session.execute(text("SELECT 1")).fetchone()
    assert result[0] == 1
    svc.engine.dispose()


@pytest.mark.asyncio
async def test_database_service_async_engine():
    """get_async_engine 返回 AsyncEngine，懒加载。"""
    import os

    os.environ["SEPTMUSE_DB_URL"] = "sqlite://"
    try:
        svc = DatabaseService()
        ae = svc.get_async_engine()
        from sqlalchemy.ext.asyncio import AsyncEngine

        assert isinstance(ae, AsyncEngine)
        # 懒加载: 第二次返回同一实例
        assert svc.get_async_engine() is ae
    finally:
        del os.environ["SEPTMUSE_DB_URL"]


def test_resolve_async_db_url_adds_driver():
    """_resolve_async_db_url 自动加 async driver。"""
    import os

    os.environ["SEPTMUSE_DB_URL"] = "sqlite:///test.db"
    try:
        svc = DatabaseService()
        async_url = svc._resolve_async_db_url()
        assert "aiosqlite" in async_url
    finally:
        del os.environ["SEPTMUSE_DB_URL"]


# ======================================================================
# P0-Task 5: WAL mode + busy_timeout + StaticPool + 连接池
# ======================================================================


def test_sqlite_wal_pragma_set(db_service):
    """SQLite WAL mode PRAGMA 在连接时设置。"""
    with db_service.engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).fetchone()
        assert mode[0].lower() == "wal"


def test_sqlite_busy_timeout_pragma_set(db_service):
    """SQLite busy_timeout PRAGMA 默认 5000ms。"""
    with db_service.engine.connect() as conn:
        timeout = conn.execute(text("PRAGMA busy_timeout")).fetchone()
        assert timeout[0] == 5000


def test_sqlite_synchronous_pragma_set(db_service):
    """SQLite synchronous PRAGMA 设为 NORMAL (WAL 配套, 1=NORMAL)。"""
    with db_service.engine.connect() as conn:
        sync = conn.execute(text("PRAGMA synchronous")).fetchone()
        assert sync[0] == 1  # 0=OFF, 1=NORMAL, 2=FULL


def test_memory_sqlite_uses_static_pool(monkeypatch):
    """:memory: SQLite 用 StaticPool 共享单连接 (跨线程并发检索)。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    config = MemoryConfig(db_path=":memory:")
    svc = DatabaseService(config=config)
    from sqlalchemy.pool import StaticPool

    assert isinstance(svc.engine.pool, StaticPool)
    svc.engine.dispose()


def test_custom_pragmas_override_defaults(tmp_path, monkeypatch):
    """自定义 sqlite_pragmas 覆盖默认值。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    custom_pragmas = {"journal_mode": "DELETE", "synchronous": "FULL", "busy_timeout": 10000}
    config = MemoryConfig(
        database=DatabaseConfig(db_path=str(tmp_path / "custom.db"), sqlite_pragmas=custom_pragmas)
    )
    svc = DatabaseService(config=config)
    with svc.engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).fetchone()
        sync = conn.execute(text("PRAGMA synchronous")).fetchone()
        timeout = conn.execute(text("PRAGMA busy_timeout")).fetchone()
    assert mode[0].lower() == "delete"
    assert sync[0] == 2  # 2=FULL
    assert timeout[0] == 10000
    svc.engine.dispose()
