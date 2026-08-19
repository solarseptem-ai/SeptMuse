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
"""DatabaseService — 数据库引擎管理 + 跨方言建表。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from septmuse.core.logging import get_logger
from septmuse.services.base import Service
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable

if TYPE_CHECKING:
    from septmuse.configs.base import MemoryConfig

logger = get_logger(__name__)


class DatabaseService(Service):
    """数据库服务 — 引擎管理 + 跨方言建表。"""

    name = "database_service"

    def __init__(self, config: MemoryConfig | None = None) -> None:
        from septmuse.configs.defaults import default_config

        self._config = config or default_config()
        self.database_url = self._resolve_db_url()
        self._dialect = self.database_url.split("://", 1)[0].split("+", 1)[0]

        connect_args = self._get_connect_args()
        # :memory: SQLite 需要 StaticPool
        if self._dialect == "sqlite" and self.database_url == "sqlite://":
            self.engine: Engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                poolclass=StaticPool,
                echo=False,
            )
        elif self._dialect in ("postgresql", "mysql"):
            self.engine: Engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                pool_size=self._config.database.connection_pool_size,
                max_overflow=self._config.database.connection_max_overflow,
                pool_timeout=self._config.database.connect_timeout,
                echo=False,
            )
        else:
            self.engine: Engine = create_engine(self.database_url, connect_args=connect_args, echo=False)

        # SQLite PRAGMA 事件监听
        if self._dialect == "sqlite":
            event.listen(self.engine, "connect", self._on_sqlite_connection)

        self.session_maker = sessionmaker(self.engine, expire_on_commit=False)
        self._async_engine: AsyncEngine | None = None

        # 注册 SQLModel 表
        for _model in (MemoryTable, HistoryTable, AccessLogTable):
            pass  # 导入即注册到 SQLModel.metadata

        self.set_ready()
        logger.info("database_service_ready", url=self._safe_url(), dialect=self._dialect)

    def _resolve_db_url(self) -> str:
        """解析 db_url: config.database.db_url > config.db_path 回退 SQLite。"""
        db_url = getattr(self._config.database, "db_url", None)
        if db_url:
            return db_url

        db_path = self._config.db_path
        if db_path is None or str(db_path) == "":
            # 默认 SQLite 路径
            db_path = Path.home() / ".septmuse" / "septmuse.db"
        db_path = str(db_path)

        if db_path == ":memory:":
            return "sqlite://"
        return f"sqlite:///{db_path}"

    def _get_connect_args(self) -> dict:
        """按方言返回连接参数。"""
        if self._dialect == "sqlite":
            return {"check_same_thread": False}
        return {}

    def _on_sqlite_connection(self, dbapi_connection, _connection_record) -> None:
        """SQLite 连接时设置 PRAGMA（WAL 模式提升并发读写）。"""
        pragmas: dict = self._config.database.sqlite_pragmas or {}
        cursor = dbapi_connection.cursor()
        try:
            for key, val in pragmas.items():
                cursor.execute(f"PRAGMA {key} = {val}")
        finally:
            cursor.close()

    def create_tables(self) -> None:
        """建表 + 迁移。"""
        from sqlmodel import SQLModel

        from septmuse.storage.migrations.runner import MigrationRunner

        SQLModel.metadata.create_all(self.engine)
        MigrationRunner(self.engine).run()
        logger.info("database_tables_created")

    def get_engine(self) -> Engine:
        """返回 SQLAlchemy engine。"""
        return self.engine

    def get_session_maker(self) -> sessionmaker:
        """返回 session 工厂。"""
        return self.session_maker

    def _resolve_async_db_url(self) -> str:
        """解析 async db_url — 自动加 async driver。"""
        url = self.database_url
        # 已有 async driver, 不重复加
        if "+aiosqlite" in url or "+aiomysql" in url or "+asyncpg" in url or "+psycopg" in url:
            return url
        # 加 async driver
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        if url.startswith("mysql://"):
            return url.replace("mysql://", "mysql+aiomysql://", 1)
        if url.startswith("mysql+pymysql://"):
            return url.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql+psycopg://"):
            return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        return url

    def get_async_engine(self) -> AsyncEngine:
        """返回 async engine（懒加载, 首次调用时创建）。"""
        if self._async_engine is None:
            async_url = self._resolve_async_db_url()
            self._async_engine = create_async_engine(async_url, echo=False)
            logger.info("async_engine_created", url=self._safe_url())
        return self._async_engine

    def get_dialect(self) -> str:
        """返回数据库方言名（sqlite/mysql/postgresql）。"""
        return self._dialect

    def _safe_url(self) -> str:
        """日志安全的 URL（隐藏密码）。"""
        url = self.database_url
        if "@" in url:
            scheme, rest = url.split("://", 1)
            if "@" in rest:
                creds, host = rest.split("@", 1)  # noqa: RUF059
                return f"{scheme}://***@{host}"
        return url

    async def teardown(self) -> None:
        """释放引擎资源。"""
        self.engine.dispose()
        logger.info("database_service_disposed")
