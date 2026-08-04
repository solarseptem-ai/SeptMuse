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
"""RelationalStoreFactory + Memory facade 集成测试。

验证:
- 有 SEPTMUSE_DB_URL → Memory 用 ORMMemoryStore
- 无 SEPTMUSE_DB_URL → Memory 用 ORMMemoryStore (零配置默认, DatabaseService 回退 SQLite)
- 有 SEPTMUSE_DB_URL → AsyncMemory 用 AsyncORMMemoryStore
"""

from __future__ import annotations


def test_factory_creates_orm_store(tmp_path, monkeypatch):
    """有 SEPTMUSE_DB_URL → Memory 用 ORMMemoryStore。"""
    db_file = tmp_path / "test_sync.db"
    monkeypatch.setenv("SEPTMUSE_DB_URL", f"sqlite:///{db_file}")
    # TypedMemoryStore 仍走 db_path, 设到 tmp 避免写 home 目录
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test_typed.db"))

    from septmuse import Memory
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

    mem = Memory()
    assert isinstance(mem.store, ORMMemoryStore)


def test_factory_without_db_url_uses_sqlite(tmp_path, monkeypatch):
    """无 SEPTMUSE_DB_URL → Memory 用 ORMMemoryStore (零配置默认, DatabaseService 回退 SQLite)。"""
    monkeypatch.delenv("SEPTMUSE_DB_URL", raising=False)
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test_sqlite.db"))

    from septmuse import Memory
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

    mem = Memory()
    assert isinstance(mem.store, ORMMemoryStore)


def test_factory_creates_async_orm_store(tmp_path, monkeypatch):
    """有 SEPTMUSE_DB_URL → AsyncMemory 用 AsyncORMMemoryStore。"""
    db_file = tmp_path / "test_async.db"
    monkeypatch.setenv("SEPTMUSE_DB_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test_async_typed.db"))

    from septmuse.memory.async_main import AsyncMemory
    from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore

    mem = AsyncMemory()
    assert isinstance(mem.store, AsyncORMMemoryStore)
