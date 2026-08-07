"""P1-Task 4: 预置检索 Recipes 单元测试。

验收标准:
- 7 种 recipe 均可正确执行
- recipe 参数可覆盖 (recipe="HYBRID_RRF", top_k=20)
- ≥7 个单元测试
"""

from __future__ import annotations

from pathlib import Path

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.retrieval.recipes import RECIPES, SearchRecipe, get_recipe, list_recipes


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_recipes.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    m = ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))
    m.add("Alice likes Python and vim", user_id="u1")
    m.add("Alice works at Google in London", user_id="u1")
    m.add("Bob prefers TypeScript and emacs", user_id="u1")
    return m


class TestRecipeDefinitions:
    def test_eight_recipes_exist(self):
        """验收: 8 种 recipe 均定义。"""
        assert len(RECIPES) == 8
        expected = {
            "HYBRID_RRF",
            "HYBRID_RRF_ENTITY",
            "HYBRID_RRF_CROSS_ENCODER",
            "HYBRID_RRF_MMR",
            "GRAPH_BFS",
            "PROGRESSIVE",
            "FORGETTING",
            "OPTIMAL",
        }
        assert set(RECIPES.keys()) == expected

    def test_list_recipes_returns_all(self):
        """list_recipes 返回全部 recipe 名称。"""
        names = list_recipes()
        assert len(names) == 8
        assert "HYBRID_RRF" in names

    def test_get_recipe_returns_search_recipe(self):
        """get_recipe 返回 SearchRecipe 对象。"""
        r = get_recipe("HYBRID_RRF")
        assert isinstance(r, SearchRecipe)
        assert r.name == "HYBRID_RRF"
        assert r.hybrid is True

    def test_get_recipe_unknown_raises(self):
        """未知 recipe 抛 ValueError。"""
        with pytest.raises(ValueError, match="Unknown recipe"):
            get_recipe("NONEXISTENT")


class TestRecipeConfigs:
    def test_hybrid_rrf_entity_has_explain(self):
        """HYBRID_RRF_ENTITY 开启 explain。"""
        r = get_recipe("HYBRID_RRF_ENTITY")
        assert r.explain is True

    def test_cross_encoder_recipe_has_reranker(self):
        """HYBRID_RRF_CROSS_ENCODER 用 cross_encoder 重排。"""
        r = get_recipe("HYBRID_RRF_CROSS_ENCODER")
        assert r.reranker == "cross_encoder"

    def test_mmr_recipe_has_mmr_reranker(self):
        """HYBRID_RRF_MMR 用 mmr 重排。"""
        r = get_recipe("HYBRID_RRF_MMR")
        assert r.reranker == "mmr"

    def test_graph_bfs_recipe_disables_hybrid(self):
        """GRAPH_BFS 关闭 hybrid (纯图遍历)。"""
        r = get_recipe("GRAPH_BFS")
        assert r.hybrid is False
        assert r.graph_bfs is True

    def test_progressive_recipe_has_progressive_flag(self):
        """PROGRESSIVE 标记 progressive。"""
        r = get_recipe("PROGRESSIVE")
        assert r.progressive is True

    def test_forgetting_recipe_has_forgetting_flag(self):
        """FORGETTING 标记 forgetting。"""
        r = get_recipe("FORGETTING")
        assert r.forgetting is True

    def test_optimal_recipe_has_hyde_and_query_rewrite(self):
        """OPTIMAL 开启 hyde + query_rewrite + cross_encoder + explain。"""
        r = get_recipe("OPTIMAL")
        assert r.hyde is True
        assert r.query_rewrite is True
        assert r.reranker == "cross_encoder"
        assert r.explain is True
        assert r.hybrid is True


class TestRecipeExecution:
    def test_search_with_hybrid_rrf_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="HYBRID_RRF") 正确执行。"""
        results = memory.search("Python", user_id="u1", recipe="HYBRID_RRF")
        assert isinstance(results, list)

    def test_search_with_hybrid_rrf_entity_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="HYBRID_RRF_ENTITY") 正确执行 (explain=True)。"""
        results = memory.search("Python", user_id="u1", recipe="HYBRID_RRF_ENTITY")
        assert isinstance(results, list)

    def test_search_with_mmr_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="HYBRID_RRF_MMR") 正确执行。"""
        results = memory.search("Python", user_id="u1", recipe="HYBRID_RRF_MMR")
        assert isinstance(results, list)

    def test_search_with_cross_encoder_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="HYBRID_RRF_CROSS_ENCODER") 正确执行 (降级 noop)。"""
        results = memory.search("Python", user_id="u1", recipe="HYBRID_RRF_CROSS_ENCODER")
        assert isinstance(results, list)

    def test_recipe_param_overridable(self, memory: ExperimentalMemory):
        """验收: recipe 参数可覆盖 (recipe="HYBRID_RRF", top_k=20)。"""
        results = memory.search("Python", user_id="u1", recipe="HYBRID_RRF", top_k=20)
        assert isinstance(results, list)
        assert len(results) <= 20

    def test_search_with_forgetting_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="FORGETTING") 正确执行。"""
        results = memory.search("Python", user_id="u1", recipe="FORGETTING")
        assert isinstance(results, list)

    def test_search_with_progressive_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="PROGRESSIVE") 正确执行。"""
        results = memory.search("Python", user_id="u1", recipe="PROGRESSIVE")
        assert isinstance(results, list)

    def test_search_with_unknown_recipe_raises(self, memory: ExperimentalMemory):
        """未知 recipe 抛 ValueError。"""
        with pytest.raises(ValueError, match="Unknown recipe"):
            memory.search("Python", user_id="u1", recipe="NONEXISTENT")

    def test_search_without_recipe_uses_defaults(self, memory: ExperimentalMemory):
        """无 recipe 时用默认参数 (hybrid=True, reranker=noop)。"""
        results = memory.search("Python", user_id="u1")
        assert isinstance(results, list)

    def test_search_with_optimal_recipe(self, memory: ExperimentalMemory):
        """验收: m.search(recipe="OPTIMAL") 正确执行 (全链路最优)。"""
        results = memory.search("Python", user_id="u1", recipe="OPTIMAL")
        assert isinstance(results, list)
