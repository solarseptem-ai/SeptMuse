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

import pytest


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


def test_factory_chroma_fallback_to_sqlite(tmp_path, monkeypatch):
    """chromadb 不可用时降级到 SQLAlchemyVectorStore + 日志警告 (零配置不 crash)。"""
    import sys

    db_file = tmp_path / "test_chroma_fallback.db"
    monkeypatch.setenv("SEPTMUSE_DB_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test_typed.db"))
    monkeypatch.setenv("SEPTMUSE_VECTOR_BACKEND", "chroma")

    # 模拟 chromadb 未安装: sys.modules["chromadb"] = None → import chromadb 触发 ImportError
    monkeypatch.setitem(sys.modules, "chromadb", None)

    from septmuse import Memory
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
    from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

    mem = Memory()
    assert isinstance(mem.store, ORMMemoryStore)
    assert isinstance(mem.store._vector_store, SQLAlchemyVectorStore)


def test_factory_chroma_available_uses_chroma(tmp_path, monkeypatch):
    """chromadb 可用时使用 ChromaVectorStore (自带 HNSW)。"""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pytest.skip("chromadb not installed")

    import tempfile

    chroma_dir = tempfile.mkdtemp(prefix="chroma_test_")
    monkeypatch.setenv("SEPTMUSE_CHROMA_PERSIST_PATH", chroma_dir)

    db_file = tmp_path / "test_chroma_available.db"
    monkeypatch.setenv("SEPTMUSE_DB_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test_typed.db"))
    monkeypatch.setenv("SEPTMUSE_VECTOR_BACKEND", "chroma")

    from septmuse import Memory
    from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
    from septmuse.storage.vector_stores.chroma import ChromaVectorStore

    mem = Memory()
    assert isinstance(mem.store, ORMMemoryStore)
    assert isinstance(mem.store._vector_store, ChromaVectorStore)
