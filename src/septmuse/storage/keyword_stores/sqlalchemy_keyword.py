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
"""SQLAlchemy 通用关键词索引 — 跨方言 LIKE + TF 评分。

任何 SQLAlchemy engine (SQLite/MySQL/PostgreSQL) 均可用。
文档存 keyword_docs 表, 检索在 Python 侧做分词匹配 + TF 评分。
upsert 用 DELETE + INSERT 两步模式 (跨方言兼容, 对齐 SQLAlchemyVectorStore)。
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.storage.keyword_stores.base import KeywordIndexBase

logger = get_logger(__name__)


def _tokenize(text_str: str) -> list[str]:
    """分词: 中英文混合, 英文按词、中文按字 (对齐 sqlite_bm25._tokenize)。"""
    return re.findall(r"[a-z0-9]+|[^\s\W]", text_str.lower())


class SQLAlchemyKeywordIndex(KeywordIndexBase):
    """SQLAlchemy 通用关键词索引 (跨方言, Python 侧 TF 评分)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        idx = SQLAlchemyKeywordIndex(engine)
        idx.add_docs({"m1": "hello world"})
        results = idx.retrieve("hello", limit=5)
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._create_table()
        logger.info("sqlalchemy_keyword_ready", dialect=engine.dialect.name)

    def _create_table(self) -> None:
        """建表 — 跨方言 DDL。"""
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS keyword_docs (
                    id   VARCHAR(512) PRIMARY KEY,
                    text TEXT NOT NULL
                )
            """
                )
            )
            conn.commit()

    def add_docs(self, docs: dict[str, str]) -> None:
        """添加或替换文档 (DELETE + INSERT 跨方言 upsert, 幂等)。"""
        with Session(self._engine) as session:
            for doc_id, doc_text in docs.items():
                session.execute(text("DELETE FROM keyword_docs WHERE id = :id").bindparams(id=doc_id))
                session.execute(
                    text("INSERT INTO keyword_docs (id, text) VALUES (:id, :text)").bindparams(
                        id=doc_id, text=doc_text
                    )
                )
            session.commit()

    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        """检索: Python 侧 TF 评分, 返回 {doc_id: score} (score [0,1], 越高越相关)。"""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT id, text FROM keyword_docs")).fetchall()
        qset = set(query_tokens)
        scores: dict[str, float] = {}
        for doc_id, doc_text in rows:
            doc_tokens = set(_tokenize(doc_text or ""))
            matched = len(qset & doc_tokens)
            if matched > 0:
                scores[doc_id] = matched / len(qset)
        if not scores:
            return {}
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(ordered[:limit])

    def delete_docs(self, doc_ids: list[str]) -> None:
        """删除文档。不存在的 id 静默跳过。"""
        with Session(self._engine) as session:
            for doc_id in doc_ids:
                session.execute(text("DELETE FROM keyword_docs WHERE id = :id").bindparams(id=doc_id))
            session.commit()

    def clear(self) -> None:
        """清空索引。"""
        with self._engine.connect() as conn:
            conn.execute(text("DELETE FROM keyword_docs"))
            conn.commit()

    def close(self) -> None:
        """释放引擎资源。"""
        self._engine.dispose()
