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
"""rank-bm25 关键词索引 — extras=[bm25] 可选实现。

用 rank-bm25 库的 BM25Okapi, SQLite docs 表持久化文档。
对比 SQLiteBM25Index (纯 Python): rank-bm25 更成熟, 支持中文分词器注入。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from septmuse.core.logging import get_logger
from septmuse.storage.keyword_stores.base import KeywordIndexBase
from septmuse.storage.keyword_stores.sqlite_bm25 import _tokenize

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
            self.conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
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
                self.conn.execute("INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)", (doc_id, text))
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
