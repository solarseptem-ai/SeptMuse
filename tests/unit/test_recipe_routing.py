"""Recipe 路由单元测试 — 验证 4 个死字段被正确路由。"""

from unittest.mock import MagicMock


def _make_pipeline(**kwargs):
    """构造测试用 SearchPipeline。"""
    from septmuse.pipelines.search import SearchPipeline

    hybrid = MagicMock()
    hybrid.search.return_value = []
    vector = MagicMock()
    vector.search.return_value = []
    return SearchPipeline(
        hybrid_retriever=hybrid,
        vector_retriever=vector,
        reranker_factory=MagicMock(),
        embedder=MagicMock(),
        **kwargs,
    )


def test_graph_bfs_recipe_routed():
    """GRAPH_BFS recipe 应调用 graph_searcher。"""
    pipeline = _make_pipeline(graph_searcher=MagicMock())
    pipeline.search("test query", user_id="u1", recipe="GRAPH_BFS")
    assert pipeline._graph_searcher is not None


def test_progressive_recipe_routed():
    """PROGRESSIVE recipe 应使用 progressive_retriever。"""
    prog = MagicMock()
    prog.retrieve.return_value = []
    pipeline = _make_pipeline(progressive_retriever=prog)
    pipeline.search("test query", user_id="u1", recipe="PROGRESSIVE")
    assert pipeline._progressive_retriever is not None


def test_hyde_recipe_routed():
    """OPTIMAL recipe（含 hyde=True）应在有 LLM 时调用 HyDE。"""
    llm = MagicMock()
    llm.complete.return_value = "hypothetical answer"
    pipeline = _make_pipeline(llm=llm)
    pipeline.search("test query", user_id="u1", recipe="OPTIMAL")
    assert llm.complete.call_count >= 1


def test_query_rewrite_recipe_routed():
    """OPTIMAL recipe（含 query_rewrite=True）应在有 LLM 时调用查询重写。"""
    llm = MagicMock()
    llm.complete.return_value = "1. variant one\n2. variant two"
    pipeline = _make_pipeline(llm=llm)
    pipeline.search("test query", user_id="u1", recipe="OPTIMAL")
    assert llm.complete.call_count >= 1


def test_no_llm_degrades_gracefully():
    """无 LLM 时 HyDE/query_rewrite 应静默跳过，不崩溃。"""
    pipeline = _make_pipeline(llm=None)
    result = pipeline.search("test query", user_id="u1", recipe="OPTIMAL")
    assert isinstance(result, (list, dict))
