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
"""BM25 自适应 sigmoid 归一化测试 (对齐 mem0 normalize_bm25).

覆盖:
- get_bm25_params 按查询词数分段返回 (midpoint, steepness)
- normalize_bm25 sigmoid 曲线 (低分压低 / midpoint=0.5 / 高分压顶)
- score_and_rank 加性融合 + threshold 过滤 + explain
- SQLiteBM25Index.retrieve 返回 [0,1] 范围分数
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from septmuse.retrieval.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)
from septmuse.storage.keyword_stores.sqlite_bm25 import SQLiteBM25Index


class TestGetBm25Params:
    """get_bm25_params 按词数分段返回参数。"""

    def test_short_query_1_word(self):
        # 1 词 → <=3 分段
        assert get_bm25_params("hello") == (5.0, 0.7)

    def test_medium_query_5_words(self):
        # 5 词 → <=6 分段
        assert get_bm25_params("hello world foo bar baz") == (7.0, 0.6)

    def test_long_query_10_words(self):
        # 10 词 → <=15 分段 (注意: split 后 10 词)
        assert get_bm25_params("hello world foo bar baz a b c d e") == (10.0, 0.5)

    def test_explicit_num_terms_overrides_split(self):
        # 显式 num_terms=2 → <=3 分段, 不走 split
        assert get_bm25_params("hello world", num_terms=2) == (5.0, 0.7)

    def test_explicit_num_terms_7_words(self):
        # 显式 num_terms=7 → <=9 分段
        assert get_bm25_params("hello", num_terms=7) == (9.0, 0.5)

    def test_very_long_query_16_words(self):
        # 16 词 → >15 分段
        words = " ".join(f"w{i}" for i in range(16))
        assert get_bm25_params(words) == (12.0, 0.5)

    def test_empty_query_defaults_to_1_term(self):
        # 空查询 → num_terms=1 → <=3 分段
        assert get_bm25_params("") == (5.0, 0.7)

    def test_boundary_3_words(self):
        # 3 词 → 边界, <=3 分段
        assert get_bm25_params("a b c") == (5.0, 0.7)

    def test_boundary_4_words(self):
        # 4 词 → <=6 分段
        assert get_bm25_params("a b c d") == (7.0, 0.6)

    def test_boundary_6_words(self):
        # 6 词 → 边界, <=6 分段
        assert get_bm25_params("a b c d e f") == (7.0, 0.6)

    def test_boundary_9_words(self):
        # 9 词 → 边界, <=9 分段
        assert get_bm25_params("a b c d e f g h i") == (9.0, 0.5)


class TestNormalizeBm25:
    """normalize_bm25 sigmoid 曲线行为。"""

    def test_zero_score_low(self):
        # raw=0, midpoint=5 → 远低于 midpoint, sigmoid 输出接近 0
        # f(0) = 1/(1+exp(3.5)) ≈ 0.0293 (低分压低)
        result = normalize_bm25(0.0, 5.0, 0.7)
        assert result == pytest.approx(1.0 / (1.0 + math.exp(0.7 * 5.0)), abs=1e-6)
        assert result < 0.05  # 远低于 0.5, 低分压低

    def test_midpoint_score_is_half(self):
        # raw=midpoint → sigmoid=0.5
        result = normalize_bm25(5.0, 5.0, 0.7)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_high_score_near_one(self):
        # raw=20 >> midpoint=5 → sigmoid ~1.0
        result = normalize_bm25(20.0, 5.0, 0.7)
        assert result > 0.9999
        assert result <= 1.0

    def test_output_range_zero_to_one(self):
        # 整条 sigmoid 曲线输出 [0, 1]
        for raw in [-10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            result = normalize_bm25(raw, 5.0, 0.7)
            assert 0.0 <= result <= 1.0, f"raw={raw} out of range: {result}"

    def test_monotonic_increasing(self):
        # sigmoid 单调递增: 原始分越高, 归一化分越高
        prev = -1.0
        for raw in [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0]:
            cur = normalize_bm25(raw, 5.0, 0.7)
            assert cur > prev, f"not monotonic at raw={raw}: {cur} <= {prev}"
            prev = cur

    def test_steepness_affects_slope(self):
        # 更陡的 steepness → 更接近阶跃函数
        gentle = normalize_bm25(5.5, 5.0, 0.1)
        steep = normalize_bm25(5.5, 5.0, 5.0)
        # 陡的离 0.5 更远 (更决断)
        assert abs(steep - 0.5) > abs(gentle - 0.5)


class TestScoreAndRank:
    """score_and_rank 加性融合。"""

    def test_basic_fusion_semantic_only(self):
        # 仅语义: max_possible=1.0, score=semantic
        results = [{"id": "m1", "score": 0.9, "payload": {"text": "a"}}]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=10)
        assert len(scored) == 1
        assert scored[0]["id"] == "m1"
        assert scored[0]["score"] == pytest.approx(0.9, abs=1e-6)

    def test_basic_fusion_all_three(self):
        # 语义 + BM25 + 实体: max_possible=2.5
        results = [{"id": "m1", "score": 0.8, "payload": {"text": "a"}}]
        bm25 = {"m1": 0.6}
        entity = {"m1": 0.3}
        scored = score_and_rank(results, bm25, entity, threshold=0.0, top_k=10)
        assert len(scored) == 1
        # (0.8 + 0.6 + 0.3) / 2.5 = 1.7/2.5 = 0.68
        assert scored[0]["score"] == pytest.approx(0.68, abs=1e-6)

    def test_fusion_capped_at_one(self):
        # 三路均满分 → 归一化后 min(2.5/2.5, 1.0) = 1.0
        results = [{"id": "m1", "score": 1.0}]
        bm25 = {"m1": 1.0}
        entity = {"m1": ENTITY_BOOST_WEIGHT}
        scored = score_and_rank(results, bm25, entity, threshold=0.0, top_k=10)
        assert scored[0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_threshold_filters_low_semantic(self):
        # semantic_score < threshold → 过滤
        results = [
            {"id": "m1", "score": 0.3, "payload": {}},
            {"id": "m2", "score": 0.8, "payload": {}},
        ]
        scored = score_and_rank(results, {}, {}, threshold=0.5, top_k=10)
        assert len(scored) == 1
        assert scored[0]["id"] == "m2"

    def test_threshold_boundary_included(self):
        # semantic_score == threshold → < 判定为 False, 保留 (严格小于才过滤)
        results = [{"id": "m1", "score": 0.5}]
        scored = score_and_rank(results, {}, {}, threshold=0.5, top_k=10)
        assert len(scored) == 1
        assert scored[0]["id"] == "m1"

    def test_explain_includes_score_details(self):
        results = [{"id": "m1", "score": 0.8, "payload": {"k": "v"}}]
        bm25 = {"m1": 0.6}
        entity = {"m1": 0.2}
        scored = score_and_rank(results, bm25, entity, threshold=0.0, top_k=10, explain=True)
        assert len(scored) == 1
        details = scored[0]["score_details"]
        assert details["semantic_score"] == 0.8
        assert details["bm25_score"] == 0.6
        assert details["entity_boost"] == 0.2
        assert details["raw_score"] == pytest.approx(1.6, abs=1e-6)
        assert details["max_possible_score"] == pytest.approx(2.5, abs=1e-6)
        assert details["final_score"] == scored[0]["score"]
        assert details["threshold"] == 0.0

    def test_no_explain_no_score_details(self):
        results = [{"id": "m1", "score": 0.8}]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=10, explain=False)
        assert "score_details" not in scored[0]

    def test_top_k_limit(self):
        # top_k 截断
        results = [
            {"id": f"m{i}", "score": 0.9 - i * 0.01, "payload": {}}
            for i in range(5)
        ]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=2)
        assert len(scored) == 2
        # 降序
        assert scored[0]["score"] >= scored[1]["score"]

    def test_sorted_descending(self):
        results = [
            {"id": "m2", "score": 0.5, "payload": {}},
            {"id": "m1", "score": 0.9, "payload": {}},
            {"id": "m3", "score": 0.7, "payload": {}},
        ]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=10)
        assert [s["id"] for s in scored] == ["m1", "m3", "m2"]

    def test_missing_id_skipped(self):
        # id=None 的项跳过
        results = [
            {"id": None, "score": 0.9, "payload": {}},
            {"id": "m1", "score": 0.8, "payload": {}},
        ]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=10)
        assert len(scored) == 1
        assert scored[0]["id"] == "m1"

    def test_payload_propagated(self):
        results = [{"id": "m1", "score": 0.8, "payload": {"text": "hello"}}]
        scored = score_and_rank(results, {}, {}, threshold=0.0, top_k=10)
        assert scored[0]["payload"] == {"text": "hello"}

    def test_missing_bm25_defaults_zero(self):
        # BM25 字典没有该 id → bm25_score=0
        results = [{"id": "m1", "score": 0.8}]
        scored = score_and_rank(results, {"m2": 0.5}, {}, threshold=0.0, top_k=10)
        # has_bm25=True → max_possible=2.0, score=(0.8+0)/2.0=0.4
        assert scored[0]["score"] == pytest.approx(0.4, abs=1e-6)

    def test_id_stringified(self):
        # id 非 str (如 int) → 转 str 索引
        results = [{"id": 123, "score": 0.8}]
        bm25 = {"123": 0.6}
        scored = score_and_rank(results, bm25, {}, threshold=0.0, top_k=10)
        assert scored[0]["id"] == "123"
        # (0.8 + 0.6) / 2.0 = 0.7
        assert scored[0]["score"] == pytest.approx(0.7, abs=1e-6)


class TestSQLiteBM25Sigmoid:
    """SQLiteBM25Index.retrieve 返回 [0,1] sigmoid 归一化分数。"""

    def test_retrieve_scores_in_unit_range(self, tmp_path: Path):
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs(
                {
                    "m1": "the quick brown fox jumps over the lazy dog",
                    "m2": "hello world python programming language",
                    "m3": "the fox is quick and brown",
                }
            )
            results = idx.retrieve("quick brown fox", limit=5)
            assert len(results) > 0
            for _doc_id, score in results.items():
                assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"
        finally:
            idx.close()

    def test_retrieve_empty_query_returns_empty(self, tmp_path: Path):
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs({"m1": "hello world"})
            assert idx.retrieve("", limit=5) == {}
        finally:
            idx.close()

    def test_retrieve_no_match_returns_empty(self, tmp_path: Path):
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs({"m1": "hello world"})
            assert idx.retrieve("zzznomatch", limit=5) == {}
        finally:
            idx.close()

    def test_retrieve_uses_sigmoid_not_max_division(self, tmp_path: Path):
        """验证用的是 sigmoid 而非 score/max_score。

        用 patch 拦截 normalize_bm25, 确认被调用且参数来自 get_bm25_params。
        """
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs(
                {
                    "m1": "alpha beta gamma",
                    "m2": "alpha beta delta",
                }
            )
            captured: dict[str, object] = {}

            def fake_normalize(raw_score: float, midpoint: float, steepness: float) -> float:
                captured["midpoint"] = midpoint
                captured["steepness"] = steepness
                captured["raw"] = raw_score
                # 返回一个明显区别于 score/max_score 的值
                return 0.42

            with patch(
                "septmuse.storage.keyword_stores.sqlite_bm25.normalize_bm25",
                side_effect=fake_normalize,
            ) as mock_norm:
                results = idx.retrieve("alpha beta", limit=5)
                assert mock_norm.call_count >= 1
                # 确认 get_bm25_params 的参数透传 (3 词 → midpoint=5.0, steepness=0.7)
                assert captured["midpoint"] == 5.0
                assert captured["steepness"] == 0.7
                # 所有分数被 fake_normalize 改成 0.42
                for score in results.values():
                    assert score == 0.42
        finally:
            idx.close()

    def test_retrieve_sorted_descending(self, tmp_path: Path):
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs(
                {
                    "m1": "python python python python",  # 高词频 → 高 BM25 原始分
                    "m2": "python",
                    "m3": "java and ruby",
                }
            )
            results = idx.retrieve("python", limit=5)
            scores = list(results.values())
            assert scores == sorted(scores, reverse=True)
        finally:
            idx.close()

    def test_retrieve_respects_limit(self, tmp_path: Path):
        idx = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
        try:
            idx.add_docs(
                {f"m{i}": f"common word{i}" for i in range(10)}
            )
            results = idx.retrieve("common", limit=3)
            assert len(results) <= 3
        finally:
            idx.close()
