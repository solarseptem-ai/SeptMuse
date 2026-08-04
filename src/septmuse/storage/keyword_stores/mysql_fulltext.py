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
"""MySQLFulltextIndex — MySQL 全文检索 (FULLTEXT INDEX + MATCH AGAINST)。

有 MySQL 方言时: 用 FULLTEXT INDEX + MATCH...AGAINST, 性能远超 Python TF。
无 MySQL 时: 降级为 SQLAlchemyKeywordIndex (Python 侧 TF), 日志警告。
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.storage.keyword_stores.base import KeywordIndexBase
from septmuse.storage.keyword_stores.sqlalchemy_keyword import SQLAlchemyKeywordIndex

logger = get_logger(__name__)


class MySQLFulltextIndex(KeywordIndexBase):
    """MySQL 全文检索索引 (有 MySQL 方言用 FULLTEXT, 无则降级)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://user:pass@host/db")
        idx = MySQLFulltextIndex(engine)
        idx.add_docs({"m1": "hello world"})
        results = idx.retrieve("hello", limit=5)
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._is_mysql = engine.dialect.name == "mysql"
        if self._is_mysql:
            self._create_table()
            logger.info("mysql_fulltext_ready")
        else:
            logger.warning("mysql_fulltext_not_available_fallback", dialect=engine.dialect.name)
            self._fallback = SQLAlchemyKeywordIndex(engine)

    def _create_table(self) -> None:
        """建表 + FULLTEXT 索引 (仅 MySQL)。"""
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS keyword_docs (
                    id   VARCHAR(512) PRIMARY KEY,
                    text TEXT,
                    FULLTEXT INDEX idx_text(text)
                )
            """
                )
            )
            conn.commit()

    def add_docs(self, docs: dict[str, str]) -> None:
        """添加或替换文档 (DELETE + INSERT, 幂等)。"""
        if not self._is_mysql:
            return self._fallback.add_docs(docs)
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
        """检索: MATCH AGAINST 评分, 归一化到 [0,1] (越高越相关)。"""
        if not self._is_mysql:
            return self._fallback.retrieve(query, limit)
        sql = text(
            """
            SELECT id, MATCH(text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS score
            FROM keyword_docs
            WHERE MATCH(text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
            ORDER BY score DESC
            LIMIT :limit
        """
        ).bindparams(query=query, limit=limit)
        with self._engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        if not rows:
            return {}
        max_score = max(float(r[1]) for r in rows)
        if max_score <= 0:
            return {str(r[0]): 1.0 for r in rows}
        return {str(r[0]): float(r[1]) / max_score for r in rows}

    def delete_docs(self, doc_ids: list[str]) -> None:
        """删除文档。不存在的 id 静默跳过。"""
        if not self._is_mysql:
            return self._fallback.delete_docs(doc_ids)
        with Session(self._engine) as session:
            for doc_id in doc_ids:
                session.execute(text("DELETE FROM keyword_docs WHERE id = :id").bindparams(id=doc_id))
            session.commit()

    def clear(self) -> None:
        """清空索引。"""
        if not self._is_mysql:
            return self._fallback.clear()
        with self._engine.connect() as conn:
            conn.execute(text("DELETE FROM keyword_docs"))
            conn.commit()

    def close(self) -> None:
        """释放引擎资源。"""
        if self._is_mysql:
            self._engine.dispose()
        else:
            self._fallback.close()
