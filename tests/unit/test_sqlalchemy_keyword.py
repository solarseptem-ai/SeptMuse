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
"""SQLAlchemyKeywordIndex 测试 — 通用关键词索引 (跨方言 TF 评分)。"""

import pytest
from sqlalchemy import create_engine

from septmuse.storage.keyword_stores.sqlalchemy_keyword import SQLAlchemyKeywordIndex


@pytest.fixture
def index():
    engine = create_engine("sqlite://", echo=False)
    idx = SQLAlchemyKeywordIndex(engine)
    yield idx
    idx.close()


def test_add_and_retrieve(index):
    """添加文档后能检索到。"""
    index.add_docs({"m1": "hello world", "m2": "foo bar"})
    results = index.retrieve("hello", limit=5)
    assert "m1" in results
    assert results["m1"] > 0


def test_retrieve_partial_match_ranks_higher(index):
    """全匹配分数高于部分匹配。"""
    index.add_docs({"m1": "hello world", "m2": "hello foo"})
    results = index.retrieve("hello world", limit=5)
    assert "m1" in results
    assert "m2" in results
    # m1 匹配全部 2 词, m2 只匹配 1 词 → m1 分数更高
    assert results["m1"] >= results["m2"]


def test_retrieve_no_match(index):
    """无匹配返回空。"""
    index.add_docs({"m1": "hello world"})
    results = index.retrieve("nonexistent", limit=5)
    assert results == {}


def test_retrieve_empty_query(index):
    """空查询返回空。"""
    index.add_docs({"m1": "hello world"})
    results = index.retrieve("", limit=5)
    assert results == {}


def test_add_docs_idempotent(index):
    """同 id 覆盖 (幂等)。"""
    index.add_docs({"m1": "hello world"})
    index.add_docs({"m1": "foo bar"})
    assert "m1" not in index.retrieve("hello", limit=5)
    assert "m1" in index.retrieve("foo", limit=5)


def test_delete_docs(index):
    """删除文档后检索不到。"""
    index.add_docs({"m1": "hello world", "m2": "foo bar"})
    index.delete_docs(["m1"])
    assert "m1" not in index.retrieve("hello", limit=5)
    assert "m2" in index.retrieve("foo", limit=5)


def test_delete_docs_silent_missing(index):
    """删除不存在的 id 静默跳过。"""
    index.add_docs({"m1": "hello world"})
    index.delete_docs(["nonexistent"])  # 不报错
    assert "m1" in index.retrieve("hello", limit=5)


def test_clear(index):
    """清空索引后检索为空。"""
    index.add_docs({"m1": "hello world", "m2": "foo bar"})
    index.clear()
    assert index.retrieve("hello", limit=5) == {}


def test_retrieve_limit(index):
    """limit 截断结果数。"""
    index.add_docs({f"m{i}": f"word{i} common" for i in range(10)})
    results = index.retrieve("common", limit=3)
    assert len(results) <= 3


def test_score_in_unit_range(index):
    """分数在 [0,1] 范围内。"""
    index.add_docs({"m1": "hello world", "m2": "hello foo"})
    results = index.retrieve("hello", limit=5)
    for score in results.values():
        assert 0.0 <= score <= 1.0
