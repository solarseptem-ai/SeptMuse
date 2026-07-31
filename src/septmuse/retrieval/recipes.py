"""预置检索 Recipes (借鉴 graphiti search_config_recipes.py)。

7 种预置配置, m.search(query, recipe="HYBRID_RRF_CROSS_ENCODER") 一键切换。

Recipes:
- HYBRID_RRF: 向量+BM25 RRF (当前默认)
- HYBRID_RRF_ENTITY: 三信号融合 + entity boost + explain
- HYBRID_RRF_CROSS_ENCODER: RRF + cross-encoder 重排
- HYBRID_RRF_MMR: RRF + MMR 去冗余
- GRAPH_BFS: 纯图遍历 (需 seed_memory_id)
- PROGRESSIVE: 渐进三层 (向量 → BM25 → 图遍历 RRF 融合)
- FORGETTING: 遗忘曲线加权 (需 typed_store, strength 衰减)
"""

from __future__ import annotations

from dataclasses import dataclass

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchRecipe:
    """预置检索配置 (借鉴 graphiti SearchConfigRecipe)。"""

    name: str
    hybrid: bool = True
    reranker: str = "noop"
    explain: bool = False
    graph_bfs: bool = False
    forgetting: bool = False
    progressive: bool = False
    description: str = ""


RECIPES: dict[str, SearchRecipe] = {
    "HYBRID_RRF": SearchRecipe(
        name="HYBRID_RRF",
        hybrid=True,
        reranker="noop",
        description="向量+BM25 RRF 融合 (默认)",
    ),
    "HYBRID_RRF_ENTITY": SearchRecipe(
        name="HYBRID_RRF_ENTITY",
        hybrid=True,
        reranker="noop",
        explain=True,
        description="三信号融合 + entity boost + explain 详情",
    ),
    "HYBRID_RRF_CROSS_ENCODER": SearchRecipe(
        name="HYBRID_RRF_CROSS_ENCODER",
        hybrid=True,
        reranker="cross_encoder",
        description="RRF + cross-encoder 重排",
    ),
    "HYBRID_RRF_MMR": SearchRecipe(
        name="HYBRID_RRF_MMR",
        hybrid=True,
        reranker="mmr",
        description="RRF + MMR 去冗余",
    ),
    "GRAPH_BFS": SearchRecipe(
        name="GRAPH_BFS",
        hybrid=False,
        graph_bfs=True,
        description="纯图遍历 (需 seed_memory_id)",
    ),
    "PROGRESSIVE": SearchRecipe(
        name="PROGRESSIVE",
        hybrid=True,
        reranker="mmr",
        progressive=True,
        description="渐进三层 (向量 → BM25 → 图遍历 RRF 融合)",
    ),
    "FORGETTING": SearchRecipe(
        name="FORGETTING",
        hybrid=True,
        forgetting=True,
        description="遗忘曲线加权 (strength 衰减排序)",
    ),
}


def get_recipe(name: str) -> SearchRecipe:
    """获取预置 recipe。未知 recipe 抛 ValueError。"""
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe: {name}. Available: {list(RECIPES.keys())}")
    return RECIPES[name]


def list_recipes() -> list[str]:
    """列出所有 recipe 名称。"""
    return list(RECIPES.keys())
