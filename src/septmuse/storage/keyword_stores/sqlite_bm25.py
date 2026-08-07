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
"""SQLite BM25 关键词索引 — 默认零配置实现 (纯 Python BM25)。

借鉴 ReMe keyword_index/bm25_index.py 的纯 Python BM25 实现,
用 SQLite docs 表持久化文档, 内存倒排索引加速检索。

参考模式 (实证):
- BM25 公式: ReMe BM25Index (k1=1.5, b=0.75 标准参数)
- 中文分词: jieba 可用时按词切分, 否则正则按字 (core.tokenizer)
- 归一化: sigmoid 自适应 (对齐 mem0 normalize_bm25), score/max_score 已弃用
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import Counter, defaultdict
from pathlib import Path

from septmuse.core.logging import get_logger
from septmuse.core.tokenizer import tokenize
from septmuse.retrieval.scoring import get_bm25_params, normalize_bm25
from septmuse.storage.keyword_stores.base import KeywordIndexBase

logger = get_logger(__name__)

_BM25_K1 = 1.5
_BM25_B = 0.75


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
        tokens = tokenize(text)
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
        tokens = tokenize(query)
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

        # sigmoid 归一化 (替代 score / max_score, 对齐 mem0 normalize_bm25)
        # 长查询原始 BM25 分偏高 → 提高 midpoint; 短查询偏低 → 降低 midpoint
        midpoint, steepness = get_bm25_params(query, num_terms=len(tokens))
        normalized = {
            doc_id: normalize_bm25(score, midpoint, steepness)
            for doc_id, score in scores.items()
        }
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
