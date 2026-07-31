# P1 存储抽象层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SeptMuse 补齐可插拔存储后端抽象层（VectorStoreBase + KeywordIndexBase + GraphStore 扩展），支持 SQLite/Chroma/Qdrant/pgvector 向量库 + SQLite BM25/rank-bm25 关键词索引 + SQLite/AGE/Neo4j 图存储，并启用混合检索（RRF 融合）。

**Architecture:** 在现有 MemoryStore ABC 下方插入三个 base 接口，SQLiteMemoryStore/PGVectorStore 重构为组合器（内部委托 VectorStore + KeywordIndex）。所有新后端通过 extras_require 可选安装，零配置默认不变（SQLite + HashEmbedder）。

**Tech Stack:** Python 3.10+, SQLite, numpy, pydantic, pytest, ruff（line-length 120）。可选：chromadb, qdrant-client, neo4j, rank-bm25。

## Global Constraints

- Python 3.10+，src/ layout，包名 `septmuse`
- ruff line-length 120，`from __future__ import annotations` 开头
- 文件头 Apache 2.0 license 注释（对齐全库）
- 零配置默认：SQLite + HashEmbedder 不变，新后端通过 extras 可选
- 现有 614 单元 + 23 e2e 测试零退化（回归基线）
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- 中文 docstring，英文内部注释
- 测试不跳过、不改断言、不删用例
- 所有 score 统一为相似度 [0,1] 越高越相似

## File Structure

```
src/septmuse/
  storage/
    base.py                       # 扩展 +2 方法 (keyword_search, hybrid_search)
    sqlite/store.py               # 重构为组合器
    vector/
      base.py                     # 新增 VectorStoreBase ABC
      sqlite_vec.py               # 新增 SQLiteVectorStore (默认)
      pgvector.py                 # 重构为组合器
      chroma.py                   # 新增 [chroma]
      qdrant.py                   # 新增 [qdrant]
    keyword/
      __init__.py                 # 新增模块
      base.py                     # 新增 KeywordIndexBase ABC
      sqlite_bm25.py              # 新增 SQLiteBM25Index (默认)
      rank_bm25.py                # 新增 [bm25]
    graph/
      base.py                     # 扩展 +1 方法 (delete_edge)
      sqlite.py                   # 扩展 +delete_edge 实现
      age.py                      # 扩展 +delete_edge 实现
      neo4j.py                    # 新增 [neo4j]
  configs/
    defaults.py                   # +3 backend 字段
pyproject.toml                    # +4 extras
tests/unit/
  test_vector_store_base.py       # 新增
  test_keyword_index.py           # 新增
  test_hybrid_search.py           # 新增
  test_composite_store.py         # 新增
```

---

### Task 1: VectorStoreBase ABC + SQLiteVectorStore

**Files:**
- Create: `src/septmuse/storage/vector/base.py`
- Create: `src/septmuse/storage/vector/sqlite_vec.py`
- Create: `tests/unit/test_vector_store_base.py`

**Interfaces:**
- Produces: `VectorStoreBase` ABC (insert_vectors/search_vectors/delete_vector/get_vector/list_vectors), `VectorSearchResult` dataclass, `VectorEntry` dataclass
- Consumes: 无（基础层）

- [ ] **Step 1: 写失败测试 — VectorStoreBase ABC 契约**

创建 `tests/unit/test_vector_store_base.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""VectorStoreBase ABC 契约 + SQLiteVectorStore CRUD 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from septmuse.storage.vector.base import VectorEntry, VectorSearchResult, VectorStoreBase


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        VectorStoreBase()


def test_vector_search_result_dataclass():
    r = VectorSearchResult(id="x", score=0.9, payload={"k": "v"})
    assert r.id == "x"
    assert r.score == 0.9
    assert r.payload == {"k": "v"}


def test_vector_entry_dataclass():
    e = VectorEntry(id="x", vector=[0.1, 0.2], payload={"k": "v"})
    assert e.id == "x"
    assert e.vector == [0.1, 0.2]
    assert e.payload == {"k": "v"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_store_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.storage.vector.base'`

- [ ] **Step 3: 实现 VectorStoreBase ABC**

创建 `src/septmuse/storage/vector/base.py`（license 头 + 完整实现按 spec §4.1）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""向量存储后端抽象基类 (借鉴 mem0 vector_stores/base.py, 精简为 5 方法)。

只管向量 CRUD, 不管 memories 表/history 表 (那是 MemoryStore 的职责)。
score 语义: 相似度 [0,1], 越高越相似。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """向量检索结果 (对齐 mem0 OutputData id/score/payload 三字段)。"""

    id: str
    score: float
    payload: dict[str, Any] | None = None


@dataclass
class VectorEntry:
    """向量条目。"""

    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None


class VectorStoreBase(ABC):
    """向量存储后端抽象 (借鉴 mem0 vector_stores/base.py, 精简为 5 方法)。

    实现方需保证 user_id 隔离 (通过 filters 参数)。
    """

    @abstractmethod
    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量插入向量。id 与 vector 一一对应。"""
        ...

    @abstractmethod
    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """向量检索, 返回 top_k 结果 (按 score 降序)。

        filters: payload 字段过滤, 如 {"user_id": "alice"}。
        """
        ...

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """删除向量。True=删除成功, False=不存在。"""
        ...

    @abstractmethod
    def get_vector(self, vector_id: str) -> VectorEntry | None:
        """取单条向量。不存在返回 None。"""
        ...

    @abstractmethod
    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        """列向量。filters 按 payload 字段过滤。"""
        ...
```

- [ ] **Step 4: 跑测试确认 ABC 契约通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_store_base.py -v`
Expected: 3 passed

- [ ] **Step 5: 写失败测试 — SQLiteVectorStore CRUD**

追加到 `tests/unit/test_vector_store_base.py`:

```python
import sqlite3

from septmuse.storage.vector.sqlite_vec import SQLiteVectorStore


@pytest.fixture()
def vec_store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    store = SQLiteVectorStore(conn=conn)
    yield store
    store.close()


def test_insert_and_search(vec_store):
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ids = ["m1", "m2"]
    payloads = [{"user_id": "alice"}, {"user_id": "alice"}]
    vec_store.insert_vectors(vectors, ids, payloads)

    results = vec_store.search_vectors([1.0, 0.0, 0.0], top_k=2, filters={"user_id": "alice"})
    assert len(results) == 2
    assert results[0].id == "m1"
    assert results[0].score > 0.99  # cosine ~1.0


def test_insert_dimension_mismatch_raises(vec_store):
    with pytest.raises(ValueError, match="dimension"):
        vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])


def test_delete_vector(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    assert vec_store.delete_vector("m1") is True
    assert vec_store.delete_vector("m1") is False  # 已删除


def test_get_vector(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    entry = vec_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert entry.payload == {"user_id": "u"}
    assert vec_store.get_vector("missing") is None


def test_list_vectors_with_filter(vec_store):
    vec_store.insert_vectors(
        [[1.0, 0.0], [0.0, 1.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    alice = vec_store.list_vectors(filters={"user_id": "alice"})
    assert len(alice) == 1
    assert alice[0].id == "m1"


def test_search_filters_by_payload(vec_store):
    vec_store.insert_vectors(
        [[1.0, 0.0], [1.0, 0.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    results = vec_store.search_vectors([1.0, 0.0], top_k=10, filters={"user_id": "bob"})
    assert len(results) == 1
    assert results[0].id == "m2"


def test_search_empty_store_returns_empty(vec_store):
    results = vec_store.search_vectors([1.0, 0.0], top_k=5)
    assert results == []


def test_search_zero_vector_returns_empty(vec_store):
    vec_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    results = vec_store.search_vectors([0.0, 0.0], top_k=5)
    assert results == []
```

- [ ] **Step 6: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_store_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.storage.vector.sqlite_vec'`

- [ ] **Step 7: 实现 SQLiteVectorStore**

创建 `src/septmuse/storage/vector/sqlite_vec.py`（license 头 + 完整实现）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""SQLite 向量存储 — 默认零配置实现 (numpy 余弦相似)。

借鉴 mem0 vector_stores/faiss.py 的 numpy 余弦回退模式,
用 SQLite vector_entries 表持久化, JSON list[float] 存向量。

参考模式 (实证):
- numpy 余弦相似: mem0 FaissVectorStore.search 的 fallback 路径
- payload JSON 列: mem0 Qdrant.payload 字段
- user_id 隔离: mem0 search filters 参数
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

import numpy as np

from septmuse.observability import get_logger
from septmuse.storage.vector.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class SQLiteVectorStore(VectorStoreBase):
    """SQLite 向量存储 (numpy 余弦, 零配置默认)。

    用法:
        conn = sqlite3.connect("mem.db")
        store = SQLiteVectorStore(conn=conn)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.conn = conn
        self._lock = lock or threading.Lock()
        self._create_table()
        logger.info("sqlite_vector_store_ready")

    def _create_table(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id       TEXT PRIMARY KEY,
                    vector   TEXT NOT NULL,
                    payload  TEXT DEFAULT '{}'
                )
                """
            )
            self.conn.commit()

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) length mismatch")
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError(f"payloads ({len(payloads)}) and ids ({len(ids)}) length mismatch")
        if payloads is None:
            payloads = [{}] * len(ids)

        with self._lock:
            for vec, vid, payload in zip(vectors, ids, payloads):
                self.conn.execute(
                    "INSERT OR REPLACE INTO vector_entries (id, vector, payload) VALUES (?, ?, ?)",
                    (vid, json.dumps(vec), json.dumps(payload)),
                )
            self.conn.commit()

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

        with self._lock:
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
        if not filters:
            return list(self.conn.execute("SELECT id, vector, payload FROM vector_entries"))
        rows = list(self.conn.execute("SELECT id, vector, payload FROM vector_entries"))
        result = []
        for vid, vec_json, payload_json in rows:
            payload = json.loads(payload_json) if payload_json else {}
            if all(payload.get(k) == v for k, v in filters.items()):
                result.append((vid, vec_json, payload_json))
        return result

    def delete_vector(self, vector_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM vector_entries WHERE id = ?", (vector_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT vector, payload FROM vector_entries WHERE id = ?", (vector_id,)
            ).fetchone()
        if row is None:
            return None
        vec_json, payload_json = row
        return VectorEntry(
            id=vector_id,
            vector=json.loads(vec_json),
            payload=json.loads(payload_json) if payload_json else {},
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
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
        self.conn.close()
```

- [ ] **Step 8: 跑测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_vector_store_base.py -v`
Expected: 12 passed

- [ ] **Step 9: ruff 检查**

Run: `ruff check src/septmuse/storage/vector/ tests/unit/test_vector_store_base.py`
Expected: All checks passed

- [ ] **Step 10: 跑全量回归确认零退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 614 + 12 = 626 passed, 9 skipped

- [ ] **Step 11: Commit**

```bash
git add src/septmuse/storage/vector/base.py src/septmuse/storage/vector/sqlite_vec.py tests/unit/test_vector_store_base.py
git commit -m "feat(storage): add VectorStoreBase ABC + SQLiteVectorStore default impl"
```

---

### Task 2: KeywordIndexBase ABC + SQLiteBM25Index

**Files:**
- Create: `src/septmuse/storage/keyword/__init__.py`
- Create: `src/septmuse/storage/keyword/base.py`
- Create: `src/septmuse/storage/keyword/sqlite_bm25.py`
- Create: `tests/unit/test_keyword_index.py`

**Interfaces:**
- Produces: `KeywordIndexBase` ABC (add_docs/retrieve/delete_docs/clear)
- Consumes: 无

- [ ] **Step 1: 写失败测试 — KeywordIndexBase 契约**

创建 `tests/unit/test_keyword_index.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""KeywordIndexBase ABC 契约 + SQLiteBM25Index 测试。"""

from __future__ import annotations

import pytest

from septmuse.storage.keyword.base import KeywordIndexBase


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        KeywordIndexBase()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_keyword_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.storage.keyword'`

- [ ] **Step 3: 实现 KeywordIndexBase ABC + 模块 __init__**

创建 `src/septmuse/storage/keyword/__init__.py`（空文件，仅 license 头）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
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
"""关键词索引后端模块。"""
```

创建 `src/septmuse/storage/keyword/base.py`（按 spec §4.2）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""关键词索引后端抽象基类 (借鉴 ReMe keyword_index/base_keyword_index.py, 改同步)。

score 语义: 归一化 BM25 分数 [0,1], 越高越相关 (与 VectorStoreBase 对齐)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class KeywordIndexBase(ABC):
    """关键词索引后端抽象。

    实现方需保证:
    - add_docs 幂等 (同 id 覆盖)
    - delete_docs 静默跳过不存在的 id
    - retrieve 返回 score 已归一化
    """

    @abstractmethod
    def add_docs(self, docs: dict[str, str]) -> None:
        """添加或替换文档 (id->text)。已存在的 id 覆盖。"""
        ...

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        """检索, 返回 {doc_id: score} (按 score 降序, top_k=limit)。"""
        ...

    @abstractmethod
    def delete_docs(self, doc_ids: list[str]) -> None:
        """删除文档。不存在的 id 静默跳过。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""
        ...
```

- [ ] **Step 4: 跑测试确认 ABC 契约通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_keyword_index.py -v`
Expected: 1 passed

- [ ] **Step 5: 写失败测试 — SQLiteBM25Index CRUD**

追加到 `tests/unit/test_keyword_index.py`:

```python
from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index


@pytest.fixture()
def bm25_store(tmp_path):
    store = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
    yield store
    store.close()


def test_add_and_retrieve(bm25_store):
    bm25_store.add_docs({
        "m1": "the quick brown fox jumps over the lazy dog",
        "m2": "slow brown turtle crawls under the log",
        "m3": "fast orange fox leaps over the fence",
    })
    results = bm25_store.retrieve("quick fox", limit=2)
    assert "m1" in results
    assert results["m1"] > 0.0


def test_add_docs_overwrite(bm25_store):
    bm25_store.add_docs({"m1": "alpha beta gamma"})
    bm25_store.add_docs({"m1": "delta epsilon zeta"})
    results = bm25_store.retrieve("delta", limit=5)
    assert "m1" in results
    results_alpha = bm25_store.retrieve("alpha", limit=5)
    assert "m1" not in results_alpha  # 已被覆盖


def test_retrieve_empty_query_returns_empty(bm25_store):
    bm25_store.add_docs({"m1": "hello world"})
    assert bm25_store.retrieve("") == {}


def test_retrieve_empty_index_returns_empty(bm25_store):
    assert bm25_store.retrieve("anything") == {}


def test_delete_docs(bm25_store):
    bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    bm25_store.delete_docs(["m1"])
    results = bm25_store.retrieve("alpha", limit=5)
    assert "m1" not in results
    results_beta = bm25_store.retrieve("beta", limit=5)
    assert "m2" in results_beta


def test_delete_nonexistent_silent(bm25_store):
    bm25_store.delete_docs(["nonexistent"])  # 不报错


def test_clear(bm25_store):
    bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    bm25_store.clear()
    assert bm25_store.retrieve("alpha") == {}
    assert bm25_store.retrieve("beta") == {}


def test_retrieve_score_normalized(bm25_store):
    bm25_store.add_docs({"m1": "alpha beta gamma", "m2": "alpha beta"})
    results = bm25_store.retrieve("alpha", limit=5)
    for score in results.values():
        assert 0.0 <= score <= 1.0


def test_chinese_tokenization(bm25_store):
    bm25_store.add_docs({
        "m1": "用户喜欢快速的应用程序",
        "m2": "系统响应缓慢",
    })
    results = bm25_store.retrieve("用户应用", limit=2)
    assert "m1" in results
```

- [ ] **Step 6: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_keyword_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.storage.keyword.sqlite_bm25'`

- [ ] **Step 7: 实现 SQLiteBM25Index**

创建 `src/septmuse/storage/keyword/sqlite_bm25.py`（license 头 + 完整实现）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""SQLite BM25 关键词索引 — 默认零配置实现 (纯 Python BM25)。

借鉴 ReMe keyword_index/bm25_index.py 的纯 Python BM25 实现,
用 SQLite docs 表持久化文档, 内存倒排索引加速检索。

参考模式 (实证):
- BM25 公式: ReMe BM25Index (k1=1.5, b=0.75 标准参数)
- 中文分词按字: ReMe RegexTokenizer + HashEmbedder._tokenize 模式
- 归一化: score / max_score → [0,1]
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from pathlib import Path

from septmuse.observability import get_logger
from septmuse.storage.keyword.base import KeywordIndexBase

logger = get_logger(__name__)

_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """分词: 中英文混合, 英文按词、中文按字 (对齐 HashEmbedder._tokenize)。"""
    return re.findall(r"[a-z0-9]+|[^\s\W]", text.lower())


class SQLiteBM25Index(KeywordIndexBase):
    """SQLite BM25 索引 (纯 Python, 零配置默认)。

    用法:
        idx = SQLiteBM25Index()
        idx.add_docs({"m1": "hello world"})
        results = idx.retrieve("hello", limit=5)
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".septmuse" / "bm25.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._create_table()
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_len: dict[str, int] = {}
        self._load_index()
        logger.info("sqlite_bm25_ready", path=str(self.db_path))

    def _create_table(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    id   TEXT PRIMARY KEY,
                    text TEXT NOT NULL
                )
                """
            )
            self.conn.commit()

    def _load_index(self) -> None:
        with self._lock:
            rows = self.conn.execute("SELECT id, text FROM docs").fetchall()
        for doc_id, text in rows:
            self._index_doc(doc_id, text)

    def _index_doc(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        self._doc_len[doc_id] = len(tokens)
        tf = Counter(tokens)
        for token, count in tf.items():
            self._inverted[token][doc_id] = count

    def _remove_doc(self, doc_id: str) -> None:
        if doc_id not in self._doc_len:
            return
        del self._doc_len[doc_id]
        for token in list(self._inverted.keys()):
            if doc_id in self._inverted[token]:
                del self._inverted[token][doc_id]
                if not self._inverted[token]:
                    del self._inverted[token]

    def add_docs(self, docs: dict[str, str]) -> None:
        with self._lock:
            for doc_id, text in docs.items():
                self._remove_doc(doc_id)
                self.conn.execute(
                    "INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)",
                    (doc_id, text),
                )
                self._index_doc(doc_id, text)
            self.conn.commit()

    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        tokens = _tokenize(query)
        if not tokens:
            return {}

        n_docs = len(self._doc_len)
        if n_docs == 0:
            return {}

        avg_len = sum(self._doc_len.values()) / n_docs
        scores: dict[str, float] = defaultdict(float)

        for token in tokens:
            postings = self._inverted.get(token, {})
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                norm = 1 - _BM25_B + _BM25_B * dl / avg_len
                score = idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * norm)
                scores[doc_id] += score

        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score == 0:
            return {}

        normalized = {doc_id: score / max_score for doc_id, score in scores.items()}
        ordered = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return dict(ordered[:limit])

    def delete_docs(self, doc_ids: list[str]) -> None:
        with self._lock:
            for doc_id in doc_ids:
                self._remove_doc(doc_id)
                self.conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
            self.conn.commit()

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM docs")
            self.conn.commit()
            self._inverted.clear()
            self._doc_len.clear()

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 8: 跑测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_keyword_index.py -v`
Expected: 11 passed

- [ ] **Step 9: ruff 检查**

Run: `ruff check src/septmuse/storage/keyword/ tests/unit/test_keyword_index.py`
Expected: All checks passed

- [ ] **Step 10: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 626 + 11 = 637 passed, 9 skipped

- [ ] **Step 11: Commit**

```bash
git add src/septmuse/storage/keyword/ tests/unit/test_keyword_index.py
git commit -m "feat(storage): add KeywordIndexBase ABC + SQLiteBM25Index default impl"
```

---

### Task 3: GraphStore.delete_edge 扩展

**Files:**
- Modify: `src/septmuse/storage/graph/base.py` (+1 abstract method)
- Modify: `src/septmuse/storage/graph/sqlite.py` (+delete_edge impl)
- Modify: `src/septmuse/storage/graph/age.py` (+delete_edge impl)
- Create: `tests/unit/test_graph_delete_edge.py`

**Interfaces:**
- Produces: `GraphStore.delete_edge(edge_id) -> bool`
- Consumes: 无（扩展现有）

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_graph_delete_edge.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""GraphStore.delete_edge 测试。"""

from __future__ import annotations

import sqlite3

from septmuse.storage.graph.sqlite import SQLiteGraphStore


def _make_store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "graph.db"))
    import threading

    lock = threading.Lock()
    return SQLiteGraphStore(conn, lock)


def test_delete_edge_success(tmp_path):
    store = _make_store(tmp_path)
    edge_id = store.add_edge("m1", "m2", "related_to", 0.8)
    assert store.delete_edge(edge_id) is True
    assert store.has_edge("m1", "m2", "related_to") is False


def test_delete_edge_nonexistent_returns_false(tmp_path):
    store = _make_store(tmp_path)
    assert store.delete_edge("nonexistent-edge-id") is False


def test_delete_edge_only_removes_target(tmp_path):
    store = _make_store(tmp_path)
    e1 = store.add_edge("m1", "m2", "related_to", 0.8)
    e2 = store.add_edge("m1", "m3", "related_to", 0.5)
    assert store.delete_edge(e1) is True
    assert store.has_edge("m1", "m2", "related_to") is False
    assert store.has_edge("m1", "m3", "related_to") is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_graph_delete_edge.py -v`
Expected: FAIL — `AttributeError: 'SQLiteGraphStore' object has no attribute 'delete_edge'`

- [ ] **Step 3: 扩展 GraphStore ABC**

在 `src/septmuse/storage/graph/base.py` 的 `GraphStore` 类末尾加（close 方法前）:

```python
    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """删除边。True=删除成功, False=不存在。"""
        ...
```

- [ ] **Step 4: SQLiteGraphStore 实现 delete_edge**

在 `src/septmuse/storage/graph/sqlite.py` 的 `SQLiteGraphStore` 类中加方法:

```python
    def delete_edge(self, edge_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM memory_links WHERE id = ?", (edge_id,))
            self.conn.commit()
            return cur.rowcount > 0
```

- [ ] **Step 5: AGEGraphStore 实现 delete_edge**

读取 `src/septmuse/storage/graph/age.py`，在 `AGEGraphStore` 类中加方法（用同样的 SQL 模式，如果 AGE 用 memory_links 表；如果用图查询语言，按 AGE 的 Cypher 风格）。

Run: `ruff check src/septmuse/storage/graph/`
Expected: All checks passed

- [ ] **Step 6: 跑测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_graph_delete_edge.py -v`
Expected: 3 passed

- [ ] **Step 7: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 637 + 3 = 640 passed, 9 skipped

- [ ] **Step 8: Commit**

```bash
git add src/septmuse/storage/graph/ tests/unit/test_graph_delete_edge.py
git commit -m "feat(storage): add GraphStore.delete_edge abstract + SQLite/AGE impl"
```

---

### Task 4: MemoryStore.keyword_search + hybrid_search + RRF

**Files:**
- Modify: `src/septmuse/storage/base.py` (+2 方法默认实现 + _rrf_fuse 函数)
- Create: `tests/unit/test_hybrid_search.py`

**Interfaces:**
- Produces: `MemoryStore.keyword_search` (默认 []), `MemoryStore.hybrid_search` (默认 RRF), `_rrf_fuse` 函数
- Consumes: 无（MemoryStore 自身扩展）

- [ ] **Step 1: 写失败测试 — RRF 融合函数**

创建 `tests/unit/test_hybrid_search.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""MemoryStore.keyword_search + hybrid_search + RRF 融合测试。"""

from __future__ import annotations

from septmuse.storage.base import _rrf_fuse


def test_rrf_empty_inputs():
    assert _rrf_fuse([], []) == []


def test_rrf_vec_only():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    result = _rrf_fuse(vec, [], alpha=1.0)
    assert len(result) == 1
    assert result[0]["id"] == "m1"
    assert result[0]["score"] > 0


def test_rrf_kw_only():
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse([], kw, alpha=0.0)
    assert len(result) == 1
    assert result[0]["id"] == "m2"


def test_rrf_fuses_and_reranks():
    vec = [
        {"id": "m1", "memory": "a", "score": 0.9},
        {"id": "m2", "memory": "b", "score": 0.7},
    ]
    kw = [
        {"id": "m2", "memory": "b", "score": 0.8},
        {"id": "m3", "memory": "c", "score": 0.6},
    ]
    result = _rrf_fuse(vec, kw, alpha=0.5)
    ids = [r["id"] for r in result]
    assert set(ids) == {"m1", "m2", "m3"}
    # m2 出现在两边, RRF 应该排第一
    assert ids[0] == "m2"


def test_rrf_alpha_pure_vec():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse(vec, kw, alpha=1.0)
    assert len(result) == 1
    assert result[0]["id"] == "m1"


def test_rrf_alpha_pure_kw():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse(vec, kw, alpha=0.0)
    assert len(result) == 1
    assert result[0]["id"] == "m2"


def test_rrf_preserves_metadata():
    vec = [{"id": "m1", "memory": "alpha", "score": 0.9, "metadata": {"k": "v"}}]
    result = _rrf_fuse(vec, [], alpha=1.0)
    assert result[0]["metadata"] == {"k": "v"}
    assert result[0]["memory"] == "alpha"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_hybrid_search.py -v`
Expected: FAIL — `ImportError: cannot import name '_rrf_fuse' from 'septmuse.storage.base'`

- [ ] **Step 3: 扩展 MemoryStore + 实现 _rrf_fuse**

在 `src/septmuse/storage/base.py` 末尾加:

```python
    def keyword_search(
        self, query: str, *, user_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。默认返回空 (子类有 KeywordIndex 时覆盖)。

        返回格式同 search: [{"id", "memory", "score", "metadata", "created_at"}]
        """
        return []

    def hybrid_search(
        self, query: str, query_embedding: list[float], *, user_id: str,
        top_k: int = 5, alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索 (向量 + 关键词 RRF 融合)。

        alpha: 向量权重 [0,1]。0=纯关键词, 1=纯向量, 0.5=均衡。
        默认实现: 向量 search + 关键词 keyword_search, RRF 融合排序。
        """
        vec_results = self.search(query_embedding, user_id=user_id, top_k=top_k * 2)
        kw_results = self.keyword_search(query, user_id=user_id, top_k=top_k * 2)
        return _rrf_fuse(vec_results, kw_results, alpha=alpha)[:top_k]


def _rrf_fuse(
    vec_results: list[dict], kw_results: list[dict], *, alpha: float = 0.5, k: int = 60
) -> list[dict]:
    """RRF 融合排序 (借鉴 Cormack 2009, k=60 标准参数)。

    score = alpha * 1/(k+rank_vec) + (1-alpha) * 1/(k+rank_kw)
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for rank, r in enumerate(vec_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + alpha / (k + rank + 1)
        meta.setdefault(r["id"], r)
    for rank, r in enumerate(kw_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + (1 - alpha) / (k + rank + 1)
        meta.setdefault(r["id"], r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[mid], "score": sc} for mid, sc in ordered]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_hybrid_search.py -v`
Expected: 7 passed

- [ ] **Step 5: 写失败测试 — MemoryStore 默认行为**

追加到 `tests/unit/test_hybrid_search.py`:

```python
from septmuse.storage.base import MemoryStore


class _StubStore(MemoryStore):
    """最小 MemoryStore 实现, 用于测试 keyword_search/hybrid_search 默认行为。"""

    def add(self, content, embedding, *, user_id, agent_id=None, metadata=None):
        return "stub"

    def search(self, query_embedding, *, user_id, top_k=5, threshold=0.1):
        return [{"id": "m1", "memory": "a", "score": 0.9, "metadata": {}, "created_at": "t"}]

    def get_all(self, *, user_id):
        return []

    def get(self, memory_id):
        return None

    def delete(self, memory_id):
        pass

    def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    def get_history(self, memory_id):
        return []

    def close(self):
        pass

    def list_agents(self, user_id):
        return []

    def list_users(self, agent_id):
        return []

    def get_shared_memories(self, user_id, limit=100):
        return []


def test_keyword_search_default_returns_empty():
    store = _StubStore()
    assert store.keyword_search("query", user_id="u") == []


def test_hybrid_search_default_uses_search_and_rrf():
    store = _StubStore()
    results = store.hybrid_search("query", [0.1, 0.2], user_id="u", top_k=5)
    # StubStore.search 返回 m1, keyword_search 返回 [], RRF 融合后应只有 m1
    assert len(results) == 1
    assert results[0]["id"] == "m1"
```

- [ ] **Step 6: 跑测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_hybrid_search.py -v`
Expected: 9 passed

- [ ] **Step 7: ruff 检查**

Run: `ruff check src/septmuse/storage/base.py tests/unit/test_hybrid_search.py`
Expected: All checks passed

- [ ] **Step 8: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 640 + 9 = 649 passed, 9 skipped

- [ ] **Step 9: Commit**

```bash
git add src/septmuse/storage/base.py tests/unit/test_hybrid_search.py
git commit -m "feat(storage): add MemoryStore.keyword_search + hybrid_search + RRF fusion"
```

---

### Task 5: SQLiteCompositeStore 重构 SQLiteMemoryStore

**Files:**
- Modify: `src/septmuse/storage/sqlite/store.py` (重构为组合器)
- Create: `tests/unit/test_composite_store.py`

**Interfaces:**
- Produces: SQLiteMemoryStore 重构后仍实现 MemoryStore（旧签名不变）
- Consumes: Task 1 的 SQLiteVectorStore, Task 2 的 SQLiteBM25Index

- [ ] **Step 1: 写组合器集成测试**

创建 `tests/unit/test_composite_store.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""SQLiteCompositeStore (SQLiteMemoryStore 重构后) 集成测试。"""

from __future__ import annotations

import numpy as np
import pytest

from septmuse.storage.sqlite.store import SQLiteMemoryStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_add_returns_memory_id(store):
    mid = store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
    assert mid.startswith("mem-")


def test_search_returns_matching_memory(store):
    store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
    store.add("foo bar", [0.0, 1.0, 0.0], user_id="alice")
    results = store.search([1.0, 0.0, 0.0], user_id="alice", top_k=2)
    assert len(results) == 2
    assert results[0]["memory"] == "hello world"


def test_keyword_search_returns_matching(store):
    store.add("the quick brown fox", [1.0, 0.0], user_id="alice")
    store.add("slow turtle", [0.0, 1.0], user_id="alice")
    results = store.keyword_search("quick fox", user_id="alice", top_k=5)
    assert any(r["memory"] == "the quick brown fox" for r in results)


def test_hybrid_search_fuses_vector_and_keyword(store):
    store.add("the quick brown fox jumps", [1.0, 0.0, 0.0], user_id="alice")
    store.add("slow turtle crawls", [0.0, 1.0, 0.0], user_id="alice")
    results = store.hybrid_search("quick fox", [1.0, 0.0, 0.0], user_id="alice", top_k=2)
    assert len(results) <= 2
    if results:
        assert results[0]["memory"] == "the quick brown fox jumps"


def test_search_user_isolation(store):
    store.add("alice secret", [1.0, 0.0], user_id="alice")
    store.add("bob secret", [1.0, 0.0], user_id="bob")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5)
    assert all(r.get("user_id", "alice") == "alice" or "alice" in str(r) for r in results)
    assert len(results) == 1


def test_delete_soft_delete(store):
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    assert store.get(mid) is None
    history = store.get_history(mid)
    assert any(h.get("event") == "DELETE" for h in history)


def test_update_changes_content(store):
    mid = store.add("original", [1.0, 0.0], user_id="alice")
    store.update(mid, "updated", [0.5, 0.5], metadata={"k": "v"})
    mem = store.get(mid)
    assert mem["memory"] == "updated"


def test_add_writes_to_both_vector_store_and_keyword_index(store):
    mid = store.add("alpha beta", [1.0, 0.0], user_id="alice")
    # 验证向量层
    assert store._vector_store.get_vector(mid) is not None
    # 验证关键词层
    kw_results = store._keyword_index.retrieve("alpha", limit=5)
    assert mid in kw_results
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_composite_store.py -v`
Expected: FAIL — `_vector_store` / `_keyword_index` 属性不存在, `keyword_search` 未实现

- [ ] **Step 3: 重构 SQLiteMemoryStore 为组合器**

读取 `src/septmuse/storage/sqlite/store.py` 全文，保留所有现有方法签名。修改点：
1. `__init__` 新增 `self._vector_store = SQLiteVectorStore(conn, lock)` 和 `self._keyword_index = SQLiteBM25Index(db_path.with_suffix(".bm25.db"))`
2. `add` 方法：在原 INSERT memories 之后，加 `self._vector_store.insert_vectors(...)` 和 `self._keyword_index.add_docs(...)`
3. `search` 方法：改为委托 `self._vector_store.search_vectors(...)` + JOIN memories 补全
4. `delete` 方法：在原软删除之后，加 `self._vector_store.delete_vector(...)` 和 `self._keyword_index.delete_docs([memory_id])`
5. `update` 方法：在原 UPDATE 之后，加 `self._vector_store.insert_vectors(...)` (覆盖) 和 `self._keyword_index.add_docs(...)` (覆盖)
6. `close` 方法：加 `self._vector_store.close()` 和 `self._keyword_index.close()`
7. 覆盖 `keyword_search`：委托 `self._keyword_index.retrieve(...)` + JOIN memories
8. `hybrid_search` 用默认实现（不覆盖）

关键实现约束：
- memories 表保留 embedding 列（双写迁移，向后兼容旧库）
- search 优先走 vector_entries，但 SQLiteVectorStore 用独立表，所以 search 直接走 vector_store
- 失败回滚：add 中 vector_store 或 keyword_index 失败时，反向删除 memories 记录
- 保留旧 `get_all`/`get`/`get_history`/`list_agents`/`list_users`/`get_shared_memories` 不变（查 memories 表）

- [ ] **Step 4: 跑组合器测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_composite_store.py -v`
Expected: 8 passed

- [ ] **Step 5: 跑全量回归确认零退化（关键）**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 649 + 8 = 657 passed, 9 skipped（614 原有 + 43 新增全绿）

如果出现失败，用 `python -m pytest tests/unit/test_sqlite_store.py -v` 定位具体回归，修复后重跑。

- [ ] **Step 6: 跑 e2e 测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/ -q`
Expected: 23 passed

- [ ] **Step 7: ruff 检查**

Run: `ruff check src/septmuse/storage/sqlite/store.py tests/unit/test_composite_store.py`
Expected: All checks passed

- [ ] **Step 8: Commit**

```bash
git add src/septmuse/storage/sqlite/store.py tests/unit/test_composite_store.py
git commit -m "refactor(storage): SQLiteMemoryStore → composite (VectorStore + KeywordIndex)"
```

---

### Task 6: PGVectorCompositeStore 重构

**Files:**
- Modify: `src/septmuse/storage/vector/pgvector.py` (重构为组合器)
- 无新测试（PG 测试默认 skip，9 个 skipped 保持）

**Interfaces:**
- Produces: PGVectorStore 重构后仍实现 MemoryStore
- Consumes: Task 1 的 VectorStoreBase, Task 2 的 KeywordIndexBase

- [ ] **Step 1: 读取 pgvector.py 全文**

Run: `Read src/septmuse/storage/vector/pgvector.py`（487 行）

- [ ] **Step 2: 重构 PGVectorStore 为组合器**

修改点（与 Task 5 同构）：
1. `__init__` 新增 `self._vector_store` (用 PGVectorVectorStore，从现有 search 逻辑提取) 和 `self._keyword_index` (用 SQLiteBM25Index，PG 不内置 BM25)
2. `add`/`search`/`delete`/`update` 委托模式同 Task 5
3. 覆盖 `keyword_search`
4. 保留所有现有方法签名不变

由于 PGVectorStore 的向量操作已经用 pgvector 扩展，提取 VectorStoreBase 实现时保留原 SQL（`<=>` 余弦距离 + `max(0, 1-distance)` 归一化）。

- [ ] **Step 3: ruff 检查**

Run: `ruff check src/septmuse/storage/vector/pgvector.py`
Expected: All checks passed

- [ ] **Step 4: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 657 passed, 9 skipped（不变）

- [ ] **Step 5: Commit**

```bash
git add src/septmuse/storage/vector/pgvector.py
git commit -m "refactor(storage): PGVectorStore → composite pattern"
```

---

### Task 7: ChromaVectorStore + QdrantVectorStore (extras)

**Files:**
- Create: `src/septmuse/storage/vector/chroma.py`
- Create: `src/septmuse/storage/vector/qdrant.py`
- Create: `tests/unit/test_chroma_vector_store.py` (@pytest.mark.integration)
- Create: `tests/unit/test_qdrant_vector_store.py` (@pytest.mark.integration)

**Interfaces:**
- Produces: `ChromaVectorStore`, `QdrantVectorStore` (实现 VectorStoreBase)
- Consumes: Task 1 的 VectorStoreBase

- [ ] **Step 1: 写 ChromaVectorStore 失败测试**

创建 `tests/unit/test_chroma_vector_store.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""ChromaVectorStore 测试 (integration, 默认 skip)。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def chroma_store(tmp_path):
    pytest.importorskip("chromadb")
    from septmuse.storage.vector.chroma import ChromaVectorStore

    store = ChromaVectorStore(persist_path=str(tmp_path / "chroma"))
    yield store
    store.close()


def test_chroma_insert_and_search(chroma_store):
    chroma_store.insert_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "alice"}],
    )
    results = chroma_store.search_vectors([1.0, 0.0, 0.0], top_k=2, filters={"user_id": "alice"})
    assert len(results) == 2
    assert results[0].id == "m1"


def test_chroma_delete(chroma_store):
    chroma_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    assert chroma_store.delete_vector("m1") is True
    assert chroma_store.delete_vector("m1") is False


def test_chroma_get_vector(chroma_store):
    chroma_store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "u"}])
    entry = chroma_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"


def test_chroma_list_vectors_filter(chroma_store):
    chroma_store.insert_vectors(
        [[1.0, 0.0], [0.0, 1.0]],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    alice = chroma_store.list_vectors(filters={"user_id": "alice"})
    assert len(alice) == 1
```

- [ ] **Step 2: 实现 ChromaVectorStore**

创建 `src/septmuse/storage/vector/chroma.py`（借鉴 mem0 `vector_stores/chroma.py`，实现 VectorStoreBase 5 方法）。Chroma 的 collection.add/query/delete/get/count 模式。score 用 Chroma 返回的 distance 转 similarity（`max(0, 1 - distance)`）。

- [ ] **Step 3: 写 QdrantVectorStore 测试**

创建 `tests/unit/test_qdrant_vector_store.py`（同结构，`pytest.importorskip("qdrant_client")`）。

- [ ] **Step 4: 实现 QdrantVectorStore**

创建 `src/septmuse/storage/vector/qdrant.py`（借鉴 mem0 `vector_stores/qdrant.py`）。Qdrant 的 upsert/search/delete/get/scroll 模式。filters 用 Qdrant 的 Filter + FieldCondition。

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/septmuse/storage/vector/chroma.py src/septmuse/storage/vector/qdrant.py`
Expected: All checks passed

- [ ] **Step 6: 跑全量回归（integration 测试默认 skip）**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 657 passed, 9 + 8 (integration) = 17 skipped

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/storage/vector/chroma.py src/septmuse/storage/vector/qdrant.py tests/unit/test_chroma_vector_store.py tests/unit/test_qdrant_vector_store.py
git commit -m "feat(storage): add ChromaVectorStore + QdrantVectorStore (extras)"
```

---

### Task 8: RankBM25Index (extras)

**Files:**
- Create: `src/septmuse/storage/keyword/rank_bm25.py`
- Create: `tests/unit/test_rank_bm25_index.py` (@pytest.mark.integration)

**Interfaces:**
- Produces: `RankBM25Index` (实现 KeywordIndexBase)
- Consumes: Task 2 的 KeywordIndexBase

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_rank_bm25_index.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""RankBM25Index 测试 (integration, 默认 skip)。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def rank_bm25_store(tmp_path):
    pytest.importorskip("rank_bm25")
    from septmuse.storage.keyword.rank_bm25 import RankBM25Index

    store = RankBM25Index(db_path=tmp_path / "rank_bm25.db")
    yield store
    store.close()


def test_rank_bm25_add_and_retrieve(rank_bm25_store):
    rank_bm25_store.add_docs({
        "m1": "the quick brown fox",
        "m2": "slow turtle",
    })
    results = rank_bm25_store.retrieve("quick fox", limit=2)
    assert "m1" in results
    assert 0.0 <= results["m1"] <= 1.0


def test_rank_bm25_delete(rank_bm25_store):
    rank_bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    rank_bm25_store.delete_docs(["m1"])
    assert "m1" not in rank_bm25_store.retrieve("alpha", limit=5)


def test_rank_bm25_clear(rank_bm25_store):
    rank_bm25_store.add_docs({"m1": "alpha"})
    rank_bm25_store.clear()
    assert rank_bm25_store.retrieve("alpha") == {}
```

- [ ] **Step 2: 实现 RankBM25Index**

创建 `src/septmuse/storage/keyword/rank_bm25.py`（用 rank-bm25 库的 BM25Okapi，SQLite docs 表持久化）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""rank-bm25 关键词索引 — extras=[bm25] 可选实现。

用 rank-bm25 库的 BM25Okapi, SQLite docs 表持久化文档。
对比 SQLiteBM25Index (纯 Python): rank-bm25 更成熟, 支持中文分词器注入。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from septmuse.observability import get_logger
from septmuse.storage.keyword.base import KeywordIndexBase
from septmuse.storage.keyword.sqlite_bm25 import _tokenize

logger = get_logger(__name__)


class RankBM25Index(KeywordIndexBase):
    """rank-bm25 索引 (extras=[bm25])。

    用法:
        pip install septmuse[bm25]
        idx = RankBM25Index()
        idx.add_docs({"m1": "hello world"})
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        from rank_bm25 import BM25Okapi

        if db_path is None:
            db_path = Path.home() / ".septmuse" / "rank_bm25.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._create_table()
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._load_index()
        self._rebuild_bm25()
        logger.info("rank_bm25_ready", path=str(self.db_path))

    def _create_table(self) -> None:
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            self.conn.commit()

    def _load_index(self) -> None:
        with self._lock:
            rows = self.conn.execute("SELECT id, text FROM docs").fetchall()
        self._doc_ids = [r[0] for r in rows]
        self._corpus = [_tokenize(r[1]) for r in rows]

    def _rebuild_bm25(self) -> None:
        if not self._corpus:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._corpus)

    def add_docs(self, docs: dict[str, str]) -> None:
        with self._lock:
            for doc_id, text in docs.items():
                if doc_id in self._doc_ids:
                    idx = self._doc_ids.index(doc_id)
                    self._corpus[idx] = _tokenize(text)
                else:
                    self._doc_ids.append(doc_id)
                    self._corpus.append(_tokenize(text))
                self.conn.execute(
                    "INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)", (doc_id, text)
                )
            self.conn.commit()
        self._rebuild_bm25()

    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        if not self._bm25 or not query.strip():
            return {}
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        max_score = max((s for _, s in indexed if s > 0), default=0.0)
        if max_score == 0:
            return {}
        result = {}
        for idx, score in indexed:
            if score <= 0 or len(result) >= limit:
                break
            result[self._doc_ids[idx]] = float(score / max_score)
        return result

    def delete_docs(self, doc_ids: list[str]) -> None:
        with self._lock:
            for doc_id in doc_ids:
                if doc_id in self._doc_ids:
                    idx = self._doc_ids.index(doc_id)
                    self._doc_ids.pop(idx)
                    self._corpus.pop(idx)
                self.conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
            self.conn.commit()
        self._rebuild_bm25()

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM docs")
            self.conn.commit()
            self._doc_ids = []
            self._corpus = []
        self._bm25 = None

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 3: ruff 检查**

Run: `ruff check src/septmuse/storage/keyword/rank_bm25.py tests/unit/test_rank_bm25_index.py`
Expected: All checks passed

- [ ] **Step 4: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 657 passed, 17 + 3 = 20 skipped

- [ ] **Step 5: Commit**

```bash
git add src/septmuse/storage/keyword/rank_bm25.py tests/unit/test_rank_bm25_index.py
git commit -m "feat(storage): add RankBM25Index (extras=[bm25])"
```

---

### Task 9: Neo4jGraphStore (extras) + 配置扩展 + pyproject extras

**Files:**
- Create: `src/septmuse/storage/graph/neo4j.py`
- Modify: `src/septmuse/configs/defaults.py` (+3 backend 字段)
- Modify: `pyproject.toml` (+4 extras)
- Create: `tests/unit/test_neo4j_graph_store.py` (@pytest.mark.integration)

**Interfaces:**
- Produces: `Neo4jGraphStore` (实现 GraphStore)
- Consumes: Task 3 的 GraphStore

- [ ] **Step 1: 写 Neo4jGraphStore 失败测试**

创建 `tests/unit/test_neo4j_graph_store.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""Neo4jGraphStore 测试 (integration, 默认 skip)。"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def neo4j_store():
    if not os.getenv("SEPTMUSE_TEST_NEO4J_URI"):
        pytest.skip("Set SEPTMUSE_TEST_NEO4J_URI to run Neo4j integration tests")
    pytest.importorskip("neo4j")
    from septmuse.storage.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(
        uri=os.getenv("SEPTMUSE_TEST_NEO4J_URI"),
        username=os.getenv("SEPTMUSE_TEST_NEO4J_USER", "neo4j"),
        password=os.getenv("SEPTMUSE_TEST_NEO4J_PASSWORD", ""),
    )
    yield store
    store.close()


def test_neo4j_add_and_get_edges(neo4j_store):
    edge_id = neo4j_store.add_edge("m1", "m2", "related_to", 0.8)
    edges = neo4j_store.get_edges("m1")
    assert any(e.id == edge_id for e in edges)


def test_neo4j_delete_edge(neo4j_store):
    edge_id = neo4j_store.add_edge("m1", "m3", "related_to", 0.5)
    assert neo4j_store.delete_edge(edge_id) is True
    assert neo4j_store.delete_edge(edge_id) is False
```

- [ ] **Step 2: 实现 Neo4jGraphStore**

创建 `src/septmuse/storage/graph/neo4j.py`（借鉴 graphiti `driver/driver.py`，实现 GraphStore 6 方法）:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... Apache 2.0 license header ...
"""Neo4j 图存储后端 — extras=[neo4j] 可选实现。

借鉴 graphiti graphiti_core/driver/driver.py 的 GraphDriver 模式,
简化为 SeptMuse GraphStore 6 方法 (add_edge/get_edges/get_neighbors/has_edge/delete_edge/close)。

参考模式 (实证):
- execute_query + parameters: graphiti GraphDriver.execute_query
- MERGE 幂等: graphiti edge save 用 MERGE
- node/edge 模型: graphiti EntityNode + EntityEdge
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.observability import get_logger
from septmuse.storage.graph.base import GraphEdge, GraphStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jGraphStore(GraphStore):
    """Neo4j 图存储 (extras=[neo4j])。

    用法:
        pip install septmuse[neo4j]
        store = Neo4jGraphStore(uri="bolt://localhost:7687", username="neo4j", password="...")
        store.add_edge("m1", "m2", "related_to", 0.8)
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connectivity()
        self._create_constraints()
        logger.info("neo4j_graph_store_ready", uri=uri)

    def _verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}") from e

    def _create_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT memory_link_id IF NOT EXISTS "
                "FOR (e:MemoryLink) REQUIRE e.id IS UNIQUE"
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        score: float = 0.0,
    ) -> str:
        edge_id = f"edge-{uuid.uuid4()}"
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:Memory {id: $source_id})
                MERGE (m:Memory {id: $target_id})
                CREATE (n)-[e:MemoryLink {id: $edge_id, relation: $relation, score: $score, created_at: $ts}]->(m)
                """,
                source_id=source_id,
                target_id=target_id,
                edge_id=edge_id,
                relation=relation,
                score=score,
                ts=_utcnow_iso(),
            )
        return edge_id

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:Memory {id: $node_id})-[e:MemoryLink]->(m:Memory)
                RETURN e.id, e.relation, e.score, $node_id, m.id
                """,
                node_id=node_id,
            )
            return [
                GraphEdge(
                    id=record["e.id"],
                    source_id=record["$node_id"],
                    target_id=record["m.id"],
                    relation=record["e.relation"],
                    score=record["e.score"],
                )
                for record in result
            ]

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        rel_filter = f"WHERE e.relation = $relation" if relation else ""
        query = f"""
            MATCH (n:Memory {{id: $node_id}})-[e:MemoryLink]->(m:Memory)
            {rel_filter}
            RETURN m.id
        """
        params = {"node_id": node_id}
        if relation:
            params["relation"] = relation
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [record["m.id"] for record in result]

    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:Memory {id: $source_id})-[e:MemoryLink {relation: $relation}]->(m:Memory {id: $target_id})
                RETURN count(e) > 0 AS exists
                """,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
            )
            return result.single()["exists"]

    def delete_edge(self, edge_id: str) -> bool:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:MemoryLink {id: $edge_id}) DELETE e RETURN count(e) AS deleted",
                edge_id=edge_id,
            )
            return result.single()["deleted"] > 0

    def close(self) -> None:
        self._driver.close()
```

- [ ] **Step 3: 扩展 MemoryConfig**

在 `src/septmuse/configs/defaults.py` 的 `MemoryConfig` 类中加字段:

```python
    vector_backend: str = Field(
        default="sqlite",
        description="向量后端: sqlite(默认)/pgvector/chroma/qdrant",
    )
    keyword_backend: str = Field(
        default="sqlite_bm25",
        description="关键词后端: sqlite_bm25(默认)/rank_bm25/none",
    )
    graph_backend: str = Field(
        default="sqlite",
        description="图后端: sqlite(默认)/age/neo4j",
    )
```

在 `default_config()` 函数末尾加:

```python
    return MemoryConfig(
        db_path=db_path,
        embedder_model=os.getenv("SEPTMUSE_EMBEDDER", "all-MiniLM-L6-v2"),
        llm_provider=os.getenv("SEPTMUSE_LLM"),
        infer=os.getenv("SEPTMUSE_INFER", "false").lower() == "true",
        vector_backend=os.getenv("SEPTMUSE_VECTOR_BACKEND", "sqlite"),
        keyword_backend=os.getenv("SEPTMUSE_KEYWORD_BACKEND", "sqlite_bm25"),
        graph_backend=os.getenv("SEPTMUSE_GRAPH_BACKEND", "sqlite"),
    )
```

- [ ] **Step 4: 扩展 pyproject.toml extras**

读取 `pyproject.toml`，在 `[project.optional-dependencies]` 段加:

```toml
chroma = ["chromadb>=0.5.0"]
qdrant = ["qdrant-client>=1.9.0"]
neo4j = ["neo4j>=5.0.0"]
bm25 = ["rank-bm25>=0.2.2"]
all-backends = ["septmuse[chroma,qdrant,neo4j,bm25]"]
```

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/septmuse/storage/graph/neo4j.py src/septmuse/configs/defaults.py`
Expected: All checks passed

- [ ] **Step 6: 跑全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 657 passed, 20 + 2 = 22 skipped

- [ ] **Step 7: 验证 extras 安装路径**

Run: `pip install -e ".[chroma]" --dry-run 2>&1 | Select-String "chromadb"`
Expected: 输出包含 chromadb（或显式提示已安装）

- [ ] **Step 8: Commit**

```bash
git add src/septmuse/storage/graph/neo4j.py src/septmuse/configs/defaults.py pyproject.toml tests/unit/test_neo4j_graph_store.py
git commit -m "feat(storage): add Neo4jGraphStore (extras) + config backend fields + pyproject extras"
```

---

### Task 10: 全量验证 + 文档更新

**Files:**
- Modify: `README.md` (+存储后端配置说明)
- Modify: `CHANGELOG.md` (+P1 记录)

- [ ] **Step 1: 全量 ruff**

Run: `ruff check src/ tests/ examples/`
Expected: All checks passed

Run: `ruff format --check src/ tests/ examples/`
Expected: 133+ files unchanged（数字可能因新文件增加）

- [ ] **Step 2: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q`
Expected: 657 passed, 22 skipped

- [ ] **Step 3: e2e 测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/ -q`
Expected: 23 passed

- [ ] **Step 4: 验证零配置默认不变**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.orchestration.memory import Memory; m = Memory(); print('OK', type(m.store).__name__)"`
Expected: `OK SQLiteMemoryStore`（或 `SQLiteCompositeStore`，取决于重构后命名）

- [ ] **Step 5: 验证 MCP 启动时间**

Run: `$env:PYTHONPATH="src"; Measure-Command { python -c "from septmuse.api.mcp.server import mcp; print('ready')" }`
Expected: < 1s

- [ ] **Step 6: 更新 README.md**

在"存储后端"或"配置"小节加存储后端配置说明（参考 spec §8 和 §9）:

```markdown
### 存储后端配置

SeptMuse 支持可插拔存储后端, 零配置默认 SQLite + HashEmbedder:

| 后端类型 | 默认 | 可选 | extras |
|---------|------|------|--------|
| 向量 | SQLite (numpy 余弦) | pgvector / Chroma / Qdrant | `[chroma]` / `[qdrant]` |
| 关键词 | SQLite BM25 (纯 Python) | rank-bm25 | `[bm25]` |
| 图 | SQLite | AGE / Neo4j | `[neo4j]` |

环境变量切换:
\`\`\`bash
SEPTMUSE_VECTOR_BACKEND=chroma
SEPTMUSE_KEYWORD_BACKEND=rank_bm25
SEPTMUSE_GRAPH_BACKEND=neo4j
\`\`\`

安装全部后端: `pip install septmuse[all-backends]`
```

- [ ] **Step 7: 更新 CHANGELOG.md**

在 `[Unreleased]` 段加:

```markdown
### Added — P1 存储抽象层

- VectorStoreBase ABC (5 方法, 借鉴 mem0 精简)
- SQLiteVectorStore (默认零配置, numpy 余弦)
- ChromaVectorStore + QdrantVectorStore (extras=[chroma]/[qdrant])
- KeywordIndexBase ABC (4 方法, 借鉴 ReMe 改同步)
- SQLiteBM25Index (默认零配置, 纯 Python BM25)
- RankBM25Index (extras=[bm25])
- GraphStore.delete_edge + Neo4jGraphStore (extras=[neo4j])
- MemoryStore.keyword_search + hybrid_search (RRF 融合)
- SQLiteMemoryStore/PGVectorStore 重构为组合器模式
- MemoryConfig +3 backend 字段 (vector_backend/keyword_backend/graph_backend)
- pyproject.toml +4 extras (chroma/qdrant/neo4j/bm25) + all-backends

### Changed

- SQLiteMemoryStore 内部重构为组合器 (VectorStore + KeywordIndex), 旧签名不变
- memories 表保留 embedding 列 (双写迁移, 向后兼容)
- 全量测试 614 → 657 passed (+43 新测试), 22 skipped (integration)
```

- [ ] **Step 8: 最终 commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: update README + CHANGELOG for P1 storage abstraction"
```

---

## Self-Review

### 1. Spec coverage

| Spec 要求 | Task | 状态 |
|----------|------|------|
| VectorStoreBase 5 方法 | Task 1 | ✅ |
| SQLiteVectorStore 默认实现 | Task 1 | ✅ |
| ChromaVectorStore + QdrantVectorStore | Task 7 | ✅ |
| KeywordIndexBase 4 方法 | Task 2 | ✅ |
| SQLiteBM25Index 默认实现 | Task 2 | ✅ |
| RankBM25Index | Task 8 | ✅ |
| GraphStore.delete_edge | Task 3 | ✅ |
| Neo4jGraphStore | Task 9 | ✅ |
| MemoryStore.keyword_search + hybrid_search | Task 4 | ✅ |
| RRF 融合 | Task 4 | ✅ |
| SQLiteCompositeStore 重构 | Task 5 | ✅ |
| PGVectorCompositeStore 重构 | Task 6 | ✅ |
| MemoryConfig +3 字段 | Task 9 | ✅ |
| pyproject.toml extras | Task 9 | ✅ |
| 验证标准 8 项 | Task 10 | ✅ |

### 2. Placeholder scan

- 无 TBD/TODO
- Task 5 Step 3 和 Task 6 Step 2 描述了"修改点"而非完整代码——这是故意的，因为重构现有 376 行 + 487 行文件需要工程师读现有代码后做适配。每个修改点都明确了"加什么/改什么/委托给谁"
- Task 7 Step 2/4 和 Task 9 Step 2 给出了完整实现骨架，工程师填 Chroma/Qdrant/Neo4j SDK 调用

### 3. Type consistency

- `VectorSearchResult(id, score, payload)` — Task 1 定义，Task 5/6/7 使用 ✅
- `VectorEntry(id, vector, payload)` — Task 1 定义，Task 1/5/6/7 使用 ✅
- `KeywordIndexBase.add_docs/retrieve/delete_docs/clear` — Task 2 定义，Task 5/6/8 使用 ✅
- `GraphStore.delete_edge(edge_id) -> bool` — Task 3 定义，Task 9 使用 ✅
- `MemoryStore.keyword_search/hybrid_search` — Task 4 定义，Task 5/6 覆盖 ✅
- `_rrf_fuse(vec_results, kw_results, alpha, k)` — Task 4 定义，Task 4 使用 ✅
- `MemoryConfig.vector_backend/keyword_backend/graph_backend` — Task 9 定义 ✅

### 4. Scope check

- 10 个 Task 适合单个实施计划，每个 Task 独立可测试
- Task 5 是关键节点（614 回归基线），如果失败需暂停修复
- Task 7/8/9 的 extras 后端测试默认 skip，不阻塞主流程
