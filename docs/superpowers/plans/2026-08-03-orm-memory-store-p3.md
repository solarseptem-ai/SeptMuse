# P3 实施计划 — VectorStore 方言工厂

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 建 `create_vector_store(engine, dialect)` 工厂 + `SQLAlchemyVectorStore`（通用 JSON+numpy）+ `PgvectorVectorStore`（pgvector 扩展，降级回退），让 ORMMemoryStore 的双写组件支持多库。

**Architecture:** `SQLAlchemyVectorStore` 是通用 VectorStoreBase——用 SQLAlchemy `text()` 执行跨方言 SQL，JSON 存向量 + numpy 余弦检索。`PgvectorVectorStore` 继承它，有 pgvector 时用 `VECTOR(dim)` 列 + `<=>` 算子，无 pgvector 时降级。

**Tech Stack:** SQLAlchemy 2.0, numpy, pgvector（可选）

## Global Constraints

- PYTHONPATH=src 运行 pytest
- ruff line-length 120
- 禁止 `ruff format` — 只用 `ruff check --no-cache`
- 代码注释用中文
- 不用 git commit
- VectorStoreBase ABC 在 `storage/vector_stores/base.py`，5 个 abstractmethod
- score 统一为相似度 [0,1]
- pgvector 扩展是可选依赖，不可用时降级 + 日志警告

## 文件结构

### 新建文件
| 文件 | 职责 |
|------|------|
| `src/septmuse/storage/vector_stores/sqlalchemy_vec.py` | SQLAlchemyVectorStore — 通用 JSON+numpy |
| `src/septmuse/storage/vector_stores/pgvector_store.py` | PgvectorVectorStore — pgvector 扩展+降级 |
| `src/septmuse/storage/vector_stores/factory.py` | create_vector_store 工厂 |
| `tests/unit/test_sqlalchemy_vector_store.py` | SQLAlchemyVectorStore 测试 |
| `tests/unit/test_vector_factory.py` | 工厂分发测试 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `src/septmuse/storage/vector_stores/__init__.py` | 导出 + 工厂 |

---

### Task 1: SQLAlchemyVectorStore — 通用 VectorStoreBase

**Files:**
- Create: `src/septmuse/storage/vector_stores/sqlalchemy_vec.py`
- Test: `tests/unit/test_sqlalchemy_vector_store.py`

**Interfaces:**
- Consumes: `VectorStoreBase` ABC, `VectorSearchResult`, `VectorEntry` from `base.py`
- Produces: `SQLAlchemyVectorStore` — 任何 SQLAlchemy engine都能用的 VectorStoreBase 实现

- [ ] **Step 1: 写测试**

创建 `tests/unit/test_sqlalchemy_vector_store.py`：

```python
"""SQLAlchemyVectorStore 测试 — 通用 JSON+numpy 向量存储。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore


@pytest.fixture
def store():
    engine = create_engine("sqlite://", echo=False)
    s = SQLAlchemyVectorStore(engine)
    yield s
    s.close()


def test_insert_and_search(store):
    """插入向量后能检索到。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"
    assert results[0].score >= 0.9


def test_search_filters_by_payload(store):
    """payload 过滤生效。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    store.insert_vectors([[0.0, 1.0]], ["m2"], [{"user_id": "bob"}])
    results = store.search_vectors([1.0, 0.0], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"


def test_delete_vector(store):
    """删除向量。"""
    store.insert_vectors([[1.0]], ["m1"])
    assert store.delete_vector("m1") is True
    assert store.delete_vector("m1") is False  # 已删


def test_get_vector(store):
    """取单条向量。"""
    store.insert_vectors([[1.0, 0.5]], ["m1"], [{"topic": "test"}])
    entry = store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert entry.vector == [1.0, 0.5]
    assert entry.payload == {"topic": "test"}


def test_get_vector_not_found(store):
    """取不存在返回 None。"""
    assert store.get_vector("nonexistent") is None


def test_list_vectors(store):
    """列向量。"""
    store.insert_vectors([[1.0], [2.0]], ["m1", "m2"], [{"user_id": "a"}, {"user_id": "b"}])
    all_vecs = store.list_vectors()
    assert len(all_vecs) == 2
    filtered = store.list_vectors(filters={"user_id": "a"})
    assert len(filtered) == 1


def test_search_empty_store(store):
    """空库检索返回空列表。"""
    results = store.search_vectors([1.0], top_k=5)
    assert results == []


def test_dimension_mismatch_raises(store):
    """维度不一致报错。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"])
    with pytest.raises(ValueError, match="dimension"):
        store.search_vectors([1.0])
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_sqlalchemy_vector_store.py -v`
Expected: FAIL ImportError

- [ ] **Step 3: 实现 SQLAlchemyVectorStore**

创建 `src/septmuse/storage/vector_stores/sqlalchemy_vec.py`：

```python
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
"""SQLAlchemy 通用向量存储 — 跨方言 JSON + numpy 余弦。

任何 SQLAlchemy engine (SQLite/MySQL/PostgreSQL) 均可用。
向量以 JSON list[float] 存储, 检索用 numpy 余弦相似。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class SQLAlchemyVectorStore(VectorStoreBase):
    """SQLAlchemy 通用向量存储 (JSON + numpy 余弦, 跨方言)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        store = SQLAlchemyVectorStore(engine)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._create_table()
        logger.info("sqlalchemy_vector_store_ready", dialect=engine.dialect.name)

    def _create_table(self) -> None:
        """建表 — 跨方言 DDL。"""
        with self._engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id      VARCHAR(512) PRIMARY KEY,
                    vector  TEXT NOT NULL,
                    payload TEXT DEFAULT '{}'
                )
            """))
            conn.commit()

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) length mismatch")
        if payloads is None:
            payloads = [{}] * len(ids)
        elif len(payloads) != len(ids):
            raise ValueError(f"payloads ({len(payloads)}) and ids ({len(ids)}) length mismatch")
        with Session(self._engine) as session:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                session.execute(text(
                    "INSERT OR REPLACE INTO vector_entries (id, vector, payload) VALUES (:id, :vec, :payload)"
                ).bindparams(id=vid, vec=json.dumps(vec), payload=json.dumps(payload)))
            session.commit()

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        query = np.array(query_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []

        rows = self._fetch_rows(filters)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for vid, vec_json, payload_json in rows:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
            if vec.shape != query.shape:
                raise ValueError(
                    f"vector dimension mismatch: query={query.shape} stored={vec.shape} for id={vid}"
                )
            vec_norm = float(np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            score = max(0.0, min(1.0, score))
            scored.append((score, vid, json.loads(payload_json) if payload_json else {}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [VectorSearchResult(id=vid, score=sc, payload=pl) for sc, vid, pl in scored[:top_k]]

    def _fetch_rows(self, filters: dict[str, Any] | None) -> list[tuple[str, str, str]]:
        """取全部行, 在 Python 侧做 payload 过滤。"""
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT id, vector, payload FROM vector_entries"))
            rows = result.fetchall()
        if not filters:
            return [(r[0], r[1], r[2]) for r in rows]
        filtered = []
        for r in rows:
            payload = json.loads(r[2]) if r[2] else {}
            if all(payload.get(k) == v for k, v in filters.items()):
                filtered.append((r[0], r[1], r[2]))
        return filtered

    def delete_vector(self, vector_id: str) -> bool:
        with self._engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            conn.commit()
            return result.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT vector, payload FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            row = result.fetchone()
        if row is None:
            return None
        return VectorEntry(
            id=vector_id,
            vector=json.loads(row[0]),
            payload=json.loads(row[1]) if row[1] else {},
        )

    def list_vectors(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[VectorEntry]:
        rows = self._fetch_rows(filters)
        entries = [
            VectorEntry(
                id=vid,
                vector=json.loads(vec_json),
                payload=json.loads(payload_json) if payload_json else {},
            )
            for vid, vec_json, payload_json in rows
        ]
        if limit is not None:
            entries = entries[:limit]
        return entries

    def close(self) -> None:
        self._engine.dispose()
```

**注意**：`INSERT OR REPLACE` 是 SQLite 语法。MySQL/PG 用 `INSERT ... ON DUPLICATE KEY UPDATE` 或 `INSERT ... ON CONFLICT DO UPDATE`。改用 SQLAlchemy 的 `dialect.insert()` 构造跨方言 upsert，或简单用 `DELETE + INSERT` 两步。最简方案——用 `Session.merge()` 替代 raw SQL upsert：

```python
    def insert_vectors(self, vectors, ids, payloads=None):
        ...
        with Session(self._engine) as session:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                # 用 SQLModel 的 merge 模式 (INSERT or UPDATE)
                session.execute(text(
                    "DELETE FROM vector_entries WHERE id = :id"
                ).bindparams(id=vid))
                session.execute(text(
                    "INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec, :payload)"
                ).bindparams(id=vid, vec=json.dumps(vec), payload=json.dumps(payload)))
            session.commit()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_sqlalchemy_vector_store.py -v`
Expected: 8 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/vector_stores/sqlalchemy_vec.py tests/unit/test_sqlalchemy_vector_store.py`
Expected: All checks passed!

---

### Task 2: PgvectorVectorStore — pgvector 扩展 + 降级

**Files:**
- Create: `src/septmuse/storage/vector_stores/pgvector_store.py`
- Test: `tests/unit/test_pgvector_vector_store.py`（仅降级路径，真实 PG 测试标记 integration）

- [ ] **Step 1: 写测试（降级路径）**

创建 `tests/unit/test_pgvector_vector_store.py`：

```python
"""PgvectorVectorStore 测试 — 降级路径（无 pgvector 时用 SQLAlchemyVectorStore）。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore


@pytest.fixture
def store():
    """用 SQLite engine 测试降级路径 (pgvector 不可用 → 回退到 SQLAlchemyVectorStore)。"""
    engine = create_engine("sqlite://", echo=False)
    s = PgvectorVectorStore(engine)
    yield s
    s.close()


def test_pgvector_fallback_insert_and_search(store):
    """降级模式: 插入+检索 (和 SQLAlchemyVectorStore 行为一致)。"""
    store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"


def test_pgvector_fallback_delete(store):
    """降级模式: 删除。"""
    store.insert_vectors([[1.0]], ["m1"])
    assert store.delete_vector("m1") is True


def test_pgvector_fallback_get(store):
    """降级模式: 取单条。"""
    store.insert_vectors([[1.0, 0.5]], ["m1"], [{"topic": "test"}])
    entry = store.get_vector("m1")
    assert entry is not None
    assert entry.vector == [1.0, 0.5]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_pgvector_vector_store.py -v`
Expected: FAIL ImportError

- [ ] **Step 3: 实现 PgvectorVectorStore**

创建 `src/septmuse/storage/vector_stores/pgvector_store.py`：

```python
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
"""PgvectorVectorStore — PostgreSQL + pgvector 扩展向量存储。

有 pgvector 扩展时: 用 VECTOR(dim) 列 + <=> 余弦距离算子, 性能远超 numpy。
无 pgvector 时: 降级为 SQLAlchemyVectorStore (JSON + numpy), 日志警告。

需要: pip install pgvector (SQLAlchemy pgvector 支持)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase
from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

logger = get_logger(__name__)

# 检测 pgvector 是否可用
try:
    from pgvector.sqlalchemy import Vector  # noqa: F401

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class PgvectorVectorStore(VectorStoreBase):
    """PostgreSQL + pgvector 向量存储 (有 pgvector 时用扩展, 无则降级)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:pass@host/db")
        store = PgvectorVectorStore(engine)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    """

    def __init__(self, engine: Engine, vector_dim: int = 384) -> None:
        self._engine = engine
        self._dim = vector_dim
        self._pgvector_available = PGVECTOR_AVAILABLE and engine.dialect.name == "postgresql"

        if self._pgvector_available:
            self._init_pgvector()
            logger.info("pgvector_store_ready", dim=vector_dim)
        else:
            # 降级为 SQLAlchemyVectorStore
            logger.warning("pgvector_not_available_fallback", dialect=engine.dialect.name)
            self._fallback = SQLAlchemyVectorStore(engine)

    def _init_pgvector(self) -> None:
        """初始化 pgvector 扩展 + 建表。"""
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id      VARCHAR(512) PRIMARY KEY,
                    vector  VECTOR({self._dim}),
                    payload JSONB DEFAULT '{{}}'::jsonb
                )
            """))
            conn.commit()

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._pgvector_available:
            return self._fallback.insert_vectors(vectors, ids, payloads)
        if payloads is None:
            payloads = [{}] * len(ids)
        with self._engine.connect() as conn:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                conn.execute(text(
                    "INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec::vector, :payload::jsonb) "
                    "ON CONFLICT (id) DO UPDATE SET vector = :vec::vector, payload = :payload::jsonb"
                ).bindparams(id=vid, vec=json.dumps(vec), payload=json.dumps(payload)))
            conn.commit()

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not self._pgvector_available:
            return self._fallback.search_vectors(query_vector, top_k, filters)

        # pgvector 余弦距离: <=> 操作符
        query_str = json.dumps(query_vector)
        sql = text("""
            SELECT id, vector, payload,
                   (vector <=> :query::vector) AS distance
            FROM vector_entries
            ORDER BY vector <=> :query::vector
            LIMIT :top_k
        """).bindparams(query=query_str, top_k=top_k)

        with self._engine.connect() as conn:
            result = conn.execute(sql)
            rows = result.fetchall()

        results: list[VectorSearchResult] = []
        for row in rows:
            vid, vec_json, payload_json, distance = row
            score = max(0.0, 1.0 - float(distance))
            payload = json.loads(payload_json) if payload_json else {}
            # payload 过滤
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            results.append(VectorSearchResult(id=str(vid), score=score, payload=payload))
        return results

    def delete_vector(self, vector_id: str) -> bool:
        if not self._pgvector_available:
            return self._fallback.delete_vector(vector_id)
        with self._engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            conn.commit()
            return result.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        if not self._pgvector_available:
            return self._fallback.get_vector(vector_id)
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT vector, payload FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            row = result.fetchone()
        if row is None:
            return None
        return VectorEntry(
            id=vector_id,
            vector=json.loads(row[0]) if row[0] else [],
            payload=json.loads(row[1]) if row[1] else {},
        )

    def list_vectors(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[VectorEntry]:
        if not self._pgvector_available:
            return self._fallback.list_vectors(filters, limit)
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT id, vector, payload FROM vector_entries"))
            rows = result.fetchall()
        entries = []
        for row in rows:
            payload = json.loads(row[2]) if row[2] else {}
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            entries.append(VectorEntry(
                id=str(row[0]),
                vector=json.loads(row[1]) if row[1] else [],
                payload=payload,
            ))
        if limit is not None:
            entries = entries[:limit]
        return entries

    def close(self) -> None:
        if self._pgvector_available:
            self._engine.dispose()
        else:
            self._fallback.close()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_pgvector_vector_store.py -v`
Expected: 3 passed（降级路径）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/vector_stores/pgvector_store.py tests/unit/test_pgvector_vector_store.py`
Expected: All checks passed!

---

### Task 3: create_vector_store 工厂 + __init__.py 导出

**Files:**
- Create: `src/septmuse/storage/vector_stores/factory.py`
- Modify: `src/septmuse/storage/vector_stores/__init__.py`
- Test: `tests/unit/test_vector_factory.py`

- [ ] **Step 1: 写测试**

创建 `tests/unit/test_vector_factory.py`：

```python
"""create_vector_store 工厂测试 — 方言分发逻辑。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.vector_stores.factory import create_vector_store
from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore
from septmuse.storage.vector_stores.sqlite_vec import SQLiteVectorStore
from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore


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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_factory.py -v`
Expected: FAIL ImportError

- [ ] **Step 3: 实现工厂 + 更新 __init__.py**

创建 `src/septmuse/storage/vector_stores/factory.py`：

```python
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

SQLite → SQLiteVectorStore (现有, 原生 sqlite3 + numpy)
PostgreSQL → PgvectorVectorStore (pgvector 扩展, 降级回退)
MySQL → SQLAlchemyVectorStore (通用 JSON + numpy)
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
        from septmuse.storage.vector_stores.sqlite_vec import SQLiteVectorStore
        # SQLite 用原生 sqlite3 连接 (性能优先)
        conn = engine.raw_connection()
        return SQLiteVectorStore(conn=conn)

    if dialect == "postgresql":
        from septmuse.storage.vector_stores.pgvector_store import PgvectorVectorStore
        return PgvectorVectorStore(engine)

    if dialect == "mysql":
        from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore
        return SQLAlchemyVectorStore(engine)

    raise ValueError(f"Unsupported dialect: {dialect}")
```

更新 `src/septmuse/storage/vector_stores/__init__.py`：

```python
"""src.septmuse.storage.vector_stores package."""

from septmuse.storage.vector_stores.factory import create_vector_store

__all__ = ["create_vector_store"]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_factory.py tests/unit/test_sqlalchemy_vector_store.py tests/unit/test_pgvector_vector_store.py -v`
Expected: 15 passed (4 factory + 8 sqlalchemy + 3 pgvector)

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/vector_stores/ tests/unit/test_vector_factory.py tests/unit/test_sqlalchemy_vector_store.py tests/unit/test_pgvector_vector_store.py`
Expected: All checks passed!

---

### Task 4: 全量回归

- [ ] **Step 1: ruff 全量**

Run: `ruff check --no-cache src/septmuse/storage/vector_stores/ tests/unit/test_vector_factory.py tests/unit/test_sqlalchemy_vector_store.py tests/unit/test_pgvector_vector_store.py`
Expected: All checks passed!

- [ ] **Step 2: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=no`
Expected: 1181+ passed + 36 skipped + 13 failed（基线不变，新增 ~15 测试）
