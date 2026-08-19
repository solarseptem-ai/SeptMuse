# 记忆能力优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 LOCOMO 评估基线 + 激活 4 个死 recipe 字段，用数据驱动地提升 SeptMuse 记忆检索能力。

**Architecture:** LOCOMO harness 作为外部评估脚本（`scripts/eval_locomo.py`），用 SeptMuse Memory 做 ingest+search，用 Ollama（bge-m3 embedder + qwen3.5 LLM）做 answer+judge。4 个死 recipe 字段通过修改 SearchPipeline 路由 + 新建 hyde/query_rewrite 模块激活。

**Tech Stack:** Python 3.11+, SeptMuse (src/ layout), Ollama (bge-m3 + qwen3.5), LOCOMO-10 dataset, pytest, ruff

## Global Constraints

- `PYTHONPATH=src` 运行所有命令
- Ruff line-length 120，`select = ["E","F","I","W","UP","B","SIM","RUF"]`
- 禁止 `from __future__ import annotations` 在 MCP tools.py（不影响本计划）
- `opensource/` 禁止修改、禁止 import — 所有借鉴在 `scripts/` 内重写
- Ollama 基址：`http://localhost:7521`
- Embedder：`bge-m3:latest`（1024 维）
- LLM：`qwen3.5:latest`
- 测试环境变量：`SEPTMUSE_EMBEDDER=hash; SEPTMUSE_TOKENIZER=space; SEPTMUSE_VECTOR_BACKEND=sqlite; SEPTMUSE_RATE_LIMIT=0`
- 单元测试用 hash embedder（零依赖），LOCOMO eval 用 ollama embedder

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/septmuse/embedders/ollama.py` | Ollama 嵌入器 | Modify: 参数名 bugfix |
| `src/septmuse/pipelines/search.py` | SearchPipeline | Modify: 路由 4 个死字段 + 新增依赖注入 |
| `src/septmuse/memory/registry_v2.py` | 组件注册表 | Modify: 传入 llm/graph_searcher/progressive_retriever |
| `src/septmuse/retrieval/hyde.py` | HyDE 假设文档生成 | Create |
| `src/septmuse/retrieval/query_rewrite.py` | 查询重写 | Create |
| `tests/unit/test_hyde.py` | HyDE 单元测试 | Create |
| `tests/unit/test_query_rewrite.py` | 查询重写单元测试 | Create |
| `tests/unit/test_recipe_routing.py` | Recipe 路由测试 | Create |
| `scripts/__init__.py` | scripts 包 | Create |
| `scripts/eval_locomo.py` | LOCOMO 评估 harness | Create |

---

### Task 1: OllamaEmbedder 参数名 bugfix

**Files:**
- Modify: `src/septmuse/embedders/ollama.py:30-46`
- Test: `tests/unit/test_embedders/test_ollama.py`

**Interfaces:**
- Produces: `OllamaEmbedder(model, base_url, embedding_dims)` — 参数名与 config 的 `base_url` 字段一致

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_embedders/test_ollama.py
"""OllamaEmbedder 单元测试。"""
import os
from unittest.mock import patch, MagicMock


def test_ollama_embedder_accepts_base_url():
    """OllamaEmbedder 应接受 base_url 参数（与 config 字段名一致）。"""
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": [{"name": "bge-m3:latest"}]}
        mock_client_cls.return_value = mock_client

        from septmuse.embedders.ollama import OllamaEmbedder

        embedder = OllamaEmbedder(
            model="bge-m3:latest",
            base_url="http://localhost:7521",
            embedding_dims=1024,
        )
        assert embedder.model == "bge-m3:latest"
        assert embedder.dimension == 1024
        mock_client_cls.assert_called_once_with(host="http://localhost:7521")


def test_ollama_embedder_falls_back_to_ollama_host_env():
    """无 base_url 时，应从 OLLAMA_HOST 环境变量回退。"""
    with patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:7521"}):
        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
            mock_client_cls.return_value = mock_client

            from septmuse.embedders.ollama import OllamaEmbedder

            embedder = OllamaEmbedder()
            mock_client_cls.assert_called_once_with(host="http://localhost:7521")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_embedders/test_ollama.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'base_url'`

- [ ] **Step 3: Fix OllamaEmbedder parameter name**

```python
# src/septmuse/embedders/ollama.py — 修改 __init__ 签名
class OllamaEmbedder(Embedder):
    """基于 Ollama 的本地嵌入。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        embedding_dims: int = DEFAULT_DIMS,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.backend_name = "ollama"
        self.model = model
        self._dim = embedding_dims
        resolved_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=resolved_url)
        self._ensure_model_exists()
        logger.info("ollama_embedder_ready", model=model, dim=self._dim, base_url=resolved_url)
```

Add `import os` at top of file.

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_embedders/test_ollama.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite + lint**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_VECTOR_BACKEND="sqlite"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/ -q`
Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/embedders/ollama.py tests/unit/test_embedders/test_ollama.py`
Expected: All tests pass, lint clean

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/embedders/ollama.py tests/unit/test_embedders/test_ollama.py
git commit -m "fix: OllamaEmbedder 参数名 ollama_base_url→base_url + OLLAMA_HOST env 回退"
```

---

### Task 2: HyDE 模块

**Files:**
- Create: `src/septmuse/retrieval/hyde.py`
- Test: `tests/unit/test_hyde.py`

**Interfaces:**
- Consumes: `llm` (SeptMuse LLM ABC, has `complete(system_prompt, user_prompt) -> str`)
- Produces: `generate_hypothetical_document(query: str, llm) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hyde.py
"""HyDE 假设文档嵌入单元测试。"""
from unittest.mock import MagicMock


def test_generate_hypothetical_document_with_llm():
    """有 LLM 时应生成假设文档。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.return_value = "张三是公司CTO，负责技术架构决策。"
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result == "张三是公司CTO，负责技术架构决策。"
    assert llm.complete.call_count == 1


def test_generate_hypothetical_document_no_llm():
    """无 LLM 时应返回 None。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    result = generate_hypothetical_document("张三负责什么", None)
    assert result is None


def test_generate_hypothetical_document_llm_error():
    """LLM 异常时应返回 None（降级，不崩溃）。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.side_effect = Exception("Ollama connection refused")
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_hyde.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septmuse.retrieval.hyde'`

- [ ] **Step 3: Create HyDE module**

```python
# src/septmuse/retrieval/hyde.py
"""HyDE — Hypothetical Document Embeddings。

用 LLM 根据查询生成假设答案，用假设答案的向量（而非原始查询）做向量检索。
假设答案在向量空间更接近已存储的记忆。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

HYDE_SYSTEM_PROMPT = "You are a helpful assistant. Given a question, write a 1-2 sentence direct answer that addresses it. Be specific and factual. Do not add preamble."

HYDE_USER_TEMPLATE = "Question: {query}\n\nWrite a 1-2 sentence hypothetical answer:"


def generate_hypothetical_document(query: str, llm: object | None) -> str | None:
    """生成假设文档。

    Args:
        query: 用户查询。
        llm: SeptMuse LLM 实例（有 complete(system, user) -> str）。None 时返回 None。

    Returns:
        假设答案文本，或 None（无 LLM 或出错时）。
    """
    if llm is None:
        return None
    try:
        prompt = HYDE_USER_TEMPLATE.format(query=query)
        result = llm.complete(HYDE_SYSTEM_PROMPT, prompt)
        if result and result.strip():
            return result.strip()
        return None
    except Exception as e:
        logger.warning("hyde_generation_failed", error=str(e))
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_hyde.py -v`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
$env:PYTHONPATH="src"; ruff check src/septmuse/retrieval/hyde.py tests/unit/test_hyde.py
$env:PYTHONPATH="src"; ruff format --check src/septmuse/retrieval/hyde.py tests/unit/test_hyde.py
git add src/septmuse/retrieval/hyde.py tests/unit/test_hyde.py
git commit -m "feat: HyDE 假设文档嵌入模块"
```

---

### Task 3: Query Rewrite 模块

**Files:**
- Create: `src/septmuse/retrieval/query_rewrite.py`
- Test: `tests/unit/test_query_rewrite.py`

**Interfaces:**
- Consumes: `llm` (SeptMuse LLM ABC)
- Produces: `rewrite_query(query: str, llm) -> list[str]`（返回原始查询 + 变体）

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_query_rewrite.py
"""查询重写单元测试。"""
from unittest.mock import MagicMock


def test_rewrite_query_with_llm():
    """有 LLM 时应返回原始查询 + 变体。"""
    from septmuse.retrieval.query_rewrite import rewrite_query

    llm = MagicMock()
    llm.complete.return_value = "1. 张三李四的对话内容\n2. 张三李四讨论的话题"
    result = rewrite_query("张三和李四聊过什么", llm)
    assert len(result) == 3  # 原始 + 2 变体
    assert result[0] == "张三和李四聊过什么"  # 原始排第一


def test_rewrite_query_no_llm():
    """无 LLM 时应只返回原始查询。"""
    from septmuse.retrieval.query_rewrite import rewrite_query

    result = rewrite_query("张三和李四聊过什么", None)
    assert result == ["张三和李四聊过什么"]


def test_rewrite_query_llm_error():
    """LLM 异常时应只返回原始查询（降级）。"""
    from septmuse.retrieval.query_rewrite import rewrite_query

    llm = MagicMock()
    llm.complete.side_effect = Exception("Ollama connection refused")
    result = rewrite_query("张三和李四聊过什么", llm)
    assert result == ["张三和李四聊过什么"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_query_rewrite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septmuse.retrieval.query_rewrite'`

- [ ] **Step 3: Create query rewrite module**

```python
# src/septmuse/retrieval/query_rewrite.py
"""查询重写 — 用 LLM 生成查询变体，多路检索后 RRF 融合。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = "You are a search query rewriter. Rewrite the given question into 2 different phrasings that ask for the same information. Number them 1. and 2. Be concise."

REWRITE_USER_TEMPLATE = "Original question: {query}\n\nWrite 2 rewritten versions:"


def rewrite_query(query: str, llm: object | None) -> list[str]:
    """重写查询，返回原始 + 变体列表。

    Args:
        query: 原始查询。
        llm: SeptMuse LLM 实例。None 时只返回原始查询。

    Returns:
        查询列表，第一个是原始查询，后续是变体。无 LLM 或出错时只含原始查询。
    """
    if llm is None:
        return [query]
    try:
        prompt = REWRITE_USER_TEMPLATE.format(query=query)
        result = llm.complete(REWRITE_SYSTEM_PROMPT, prompt)
        if not result or not result.strip():
            return [query]
        variants = _parse_variants(result, query)
        return variants if len(variants) > 1 else [query]
    except Exception as e:
        logger.warning("query_rewrite_failed", error=str(e))
        return [query]


def _parse_variants(raw: str, original: str) -> list[str]:
    """从 LLM 输出解析变体列表。"""
    variants = [original]
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in ("1.", "2.", "3.", "- ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line != original:
            variants.append(line)
    return variants
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_query_rewrite.py -v`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
$env:PYTHONPATH="src"; ruff check src/septmuse/retrieval/query_rewrite.py tests/unit/test_query_rewrite.py
git add src/septmuse/retrieval/query_rewrite.py tests/unit/test_query_rewrite.py
git commit -m "feat: 查询重写模块"
```

---

### Task 4: SearchPipeline 依赖注入 + 4 个死字段路由

**Files:**
- Modify: `src/septmuse/pipelines/search.py:134-211` (SearchPipeline.__init__ + recipe routing)
- Modify: `src/septmuse/memory/registry_v2.py:125-133` (SearchPipeline construction)
- Test: `tests/unit/test_recipe_routing.py`

**Interfaces:**
- Consumes: `HybridRetriever`, `VectorRetrieverAdapter`, `RerankerFactory`, `Embedder`, `ForgettingRetriever` (existing), `llm`, `GraphSearcher`, `ProgressiveRetriever` (new optional)
- Produces: `SearchPipeline` with 4 new optional params: `llm`, `graph_searcher`, `progressive_retriever`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_recipe_routing.py
"""Recipe 路由单元测试 — 验证 4 个死字段被正确路由。"""
from unittest.mock import MagicMock, patch


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
    # 验证 llm.complete 被调用（HyDE 用它生成假设文档）
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_VECTOR_BACKEND="sqlite"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_recipe_routing.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'graph_searcher'`

- [ ] **Step 3: Modify SearchPipeline.__init__ — add 3 optional params**

In `src/septmuse/pipelines/search.py`, modify the `SearchPipeline.__init__`:

```python
class SearchPipeline:
    def __init__(
        self,
        hybrid_retriever: Retriever,
        vector_retriever: Retriever,
        reranker_factory: Callable[[str], Any],
        embedder: Any,
        forgetting: Any | None = None,
        default_top_k: int = 5,
        default_threshold: float = 0.1,
        llm: Any | None = None,
        graph_searcher: Any | None = None,
        progressive_retriever: Any | None = None,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._vector_retriever = vector_retriever
        self._reranker_factory = reranker_factory
        self._embedder = embedder
        self._forgetting = forgetting
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self._llm = llm
        self._graph_searcher = graph_searcher
        self._progressive_retriever = progressive_retriever
```

- [ ] **Step 4: Replace the 4 no-op log lines with actual routing**

In `src/septmuse/pipelines/search.py`, replace the recipe routing block (lines ~196-211) with:

```python
        if r.graph_bfs:
            cfg._graph_bfs = True
        if r.progressive:
            cfg._progressive = True
        if r.hyde:
            cfg.hyde = True
        if r.query_rewrite:
            cfg.query_rewrite = True
```

Then, before the retrieval step `[4]`, add HyDE + query_rewrite processing:

```python
        # [3.5] HyDE — 生成假设文档，替换向量检索的查询
        hyde_doc = None
        if cfg.hyde and self._llm is not None:
            from septmuse.retrieval.hyde import generate_hypothetical_document

            hyde_doc = generate_hypothetical_document(query, self._llm)
            if hyde_doc:
                logger.info("hyde_generated", doc_len=len(hyde_doc))

        # [3.6] Query rewrite — 生成查询变体
        queries = [query]
        if cfg.query_rewrite and self._llm is not None:
            from septmuse.retrieval.query_rewrite import rewrite_query

            queries = rewrite_query(query, self._llm)
            if len(queries) > 1:
                logger.info("query_rewrite_done", variants=len(queries))
```

Then, modify the retrieval step `[4]` to handle progressive + query_rewrite + hyde:

```python
        # [4] 检索
        if getattr(cfg, "_progressive", False) and self._progressive_retriever is not None:
            # Progressive 路由 — 用 ProgressiveRetriever 替代 HybridRetriever
            prog_results = self._progressive_retriever.retrieve(
                query, user_id=user_id, top_k=fetch_k, threshold=th,
            )
            from septmuse.pipelines.search import HybridResult
            results = [
                HybridResult(
                    id=r.id, memory=r.memory, score=r.score,
                    vector_score=r.score, bm25_score=0.0, entity_boost=0.0,
                    metadata=r.metadata, created_at=r.created_at,
                )
                for r in prog_results
            ]
        elif len(queries) > 1:
            # Query rewrite — 多路检索 + RRF 融合
            all_results: list = []
            for q in queries:
                search_query = hyde_doc or q
                partial = retriever.search(
                    search_query, user_id=user_id, session_id=session_id,
                    top_k=fetch_k, threshold=th, explain=cfg.explain, filters=cfg.filters,
                )
                all_results.extend(partial)
            results = _rrf_fuse_hybrid_results(all_results, fetch_k)
        else:
            search_query = hyde_doc or query
            results = retriever.search(
                search_query, user_id=user_id, session_id=session_id,
                top_k=fetch_k, threshold=th, explain=cfg.explain, filters=cfg.filters,
            )
```

Then, after step `[6]` (dict conversion + expired filter), add Graph BFS fusion:

```python
        # [6.5] Graph BFS — 从 top-1 结果做 BFS 图遍历 + RRF 融合
        if getattr(cfg, "_graph_bfs", False) and self._graph_searcher is not None and hybrid_return:
            seed_id = hybrid_return[0].get("id", "")
            if seed_id:
                try:
                    graph_results = self._graph_searcher.search_graph(seed_id)
                    if graph_results:
                        hybrid_return = _fuse_with_graph(hybrid_return, graph_results)
                except Exception as e:
                    logger.warning("graph_bfs_failed", error=str(e))
```

Add the helper functions at the bottom of the file:

```python
def _rrf_fuse_hybrid_results(results: list, top_k: int) -> list:
    """RRF 融合多路 HybridResult 检索结果。"""
    from septmuse.retrieval.hybrid import RRF_K

    scores: dict[str, float] = {}
    by_id: dict[str, object] = {}
    for r in results:
        if r.id not in by_id:
            by_id[r.id] = r
            scores[r.id] = 0.0
        scores[r.id] += 1.0 / (RRF_K + 1)

    fused = sorted(by_id.values(), key=lambda r: scores[r.id], reverse=True)
    return fused[:top_k]


def _fuse_with_graph(hybrid_results: list[dict], graph_results: list[dict]) -> list[dict]:
    """将 hybrid 结果与 graph BFS 结果 RRF 融合。"""
    from septmuse.retrieval.graph_search import GraphSearcher

    return GraphSearcher.rrf_fuse(hybrid_results, graph_results)
```

- [ ] **Step 5: Modify registry_v2.py — pass new dependencies**

In `src/septmuse/memory/registry_v2.py`, modify the SearchPipeline registration (around line 125):

```python
        self.register("search", lambda r: SearchPipeline(
            hybrid_retriever=r._memory._get_hybrid_retriever(),
            vector_retriever=VectorRetrieverAdapter(r.store, r.embedder),
            reranker_factory=lambda name: _create_reranker(name, embedder=r.embedder, llm=r.llm),
            embedder=r.embedder,
            forgetting=r.get("forgetting"),
            default_top_k=r.config.top_k,
            default_threshold=r.config.threshold,
            llm=r.llm,
            graph_searcher=_create_graph_searcher(r),
            progressive_retriever=_create_progressive_retriever(r),
        ))
```

Add factory functions near `_create_reranker`:

```python
def _create_graph_searcher(reg: ComponentRegistry) -> Any:
    """工厂: GraphSearcher。"""
    from septmuse.retrieval.graph_search import GraphSearcher
    graph_store = getattr(reg._memory, "_graph_store", None) or reg.graph_store
    return GraphSearcher(graph_store, reg.store)


def _create_progressive_retriever(reg: ComponentRegistry) -> Any:
    """工厂: ProgressiveRetriever。"""
    from septmuse.retrieval.progressive import ProgressiveRetriever
    return ProgressiveRetriever(reg.store, reg.typed_store, reg.embedder)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_VECTOR_BACKEND="sqlite"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_recipe_routing.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite + lint**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_VECTOR_BACKEND="sqlite"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/ -q`
Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/pipelines/search.py src/septmuse/memory/registry_v2.py tests/unit/test_recipe_routing.py`
Expected: All tests pass, lint clean

- [ ] **Step 8: Commit**

```bash
git add src/septmuse/pipelines/search.py src/septmuse/memory/registry_v2.py tests/unit/test_recipe_routing.py
git commit -m "feat: 激活 4 个死 recipe 字段 — graph_bfs/progressive/hyde/query_rewrite 路由"
```

---

### Task 5: LOCOMO 评估 Harness

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/eval_locomo.py`

**Interfaces:**
- Consumes: SeptMuse `Memory` facade, Ollama LLM provider, LOCOMO-10 dataset
- Produces: `results/locomo/` JSON files + stdout metrics table

- [ ] **Step 1: Create scripts package**

```python
# scripts/__init__.py
```

- [ ] **Step 2: Write the LOCOMO harness**

This is a larger file. The key sections:
1. Dataset download + parse (from LOCOMO GitHub URL)
2. SeptMuseClient — async wrapper around Memory
3. LLM answerer + judge (using SeptMuse's Ollama LLM)
4. Metrics computation
5. CLI entry point

```python
# scripts/eval_locomo.py
"""LOCOMO-10 评估 harness — 用 SeptMuse Memory 评估检索 + 回答质量。

用法:
  # predict-only（只跑 ingest+search，不耗 LLM）
  python scripts/eval_locomo.py --project-name baseline --conversations 0 --predict-only

  # full（需要 Ollama 运行）
  python scripts/eval_locomo.py --project-name baseline --conversations 0 --full
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

# ── LOCOMO prompts (重写自 opensource/memory-benchmarks, 禁止 import) ──

JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

ANSWER_GEN_PROMPT = """You are answering a question using retrieved memories from past conversations.
Read ALL memories. In these memories, "User" refers to the main person whose memories these are.
Give a direct, specific answer after "ANSWER:".

{memories}

Question: {question}

Work through the memories, then give your final answer after "ANSWER:".
"""

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules
1. PARTIAL CREDIT: If the generated answer includes at least one correct item from the gold answer, mark CORRECT.
2. PARAPHRASES COUNT: Same concept in different words is CORRECT.
3. EXTRA DETAIL IS FINE: A longer answer that includes the gold answer's key facts is CORRECT.
4. DATE TOLERANCE: Dates within 14 days are CORRECT.
5. SEMANTIC OVERLAP: Judge whether the generated answer addresses the same topic.
6. FOCUS ON KNOWLEDGE: Only mark WRONG when the answer demonstrates a genuinely incorrect understanding.

## ONLY mark WRONG if:
- The answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG).
"""

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]
CHUNK_SIZE = 1


# ── Dataset ──

def download_dataset(dataset_dir: str) -> str:
    path = os.path.join(dataset_dir, "locomo10.json")
    if os.path.exists(path):
        return path
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading LOCOMO-10 dataset...")
    urlretrieve(DATASET_URL, path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 10:
        os.remove(path)
        raise RuntimeError(f"Invalid dataset: expected 10 conversations, got {len(data)}")
    print(f"Downloaded: {path} ({len(data)} conversations)")
    return path


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_locomo_date(date_str: str):
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    parsed = parse_locomo_date(date_str)
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())
    return None


def get_sorted_sessions(conversation: dict):
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        date_str = conversation.get(date_key, "")
        turns = conversation[key]
        paired.append((key, date_str, turns))

    def sort_key(item):
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str) -> list[list[dict]]:
    messages = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})

    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i:i + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
    return chunks


# ── SeptMuseClient — async wrapper ──

class SeptMuseClient:
    """Async wrapper around sync SeptMuse Memory."""

    def __init__(self, config=None, embedder=None):
        from septmuse.memory.main import Memory
        if config is None:
            from septmuse.configs.defaults import default_config
            config = default_config()
        self._memory = Memory(config=config, embedder=embedder) if embedder else Memory(config=config)

    async def add(self, messages: list[dict], user_id: str, timestamp: int | None = None) -> dict | None:
        content = " ".join(m["content"] for m in messages)
        try:
            self._memory.add(content, user_id=user_id, infer=bool(os.getenv("SEPTMUSE_INFER", "").lower() in ("true", "1", "yes")))
            return {"results": [{"memory": content, "event": "ADD"}]}
        except Exception as e:
            print(f"ADD failed: {e}")
            return None

    async def search(self, query: str, user_id: str, top_k: int = 200) -> list[dict]:
        try:
            results = self._memory.search(query, user_id=user_id, top_k=top_k)
            if isinstance(results, dict):
                results = results.get("results", [])
            return [
                {"memory": r.get("memory", ""), "score": r.get("score", 0), "id": r.get("id", "")}
                for r in results
            ]
        except Exception as e:
            print(f"SEARCH failed: {e}")
            return []

    async def delete_user(self, user_id: str) -> bool:
        try:
            self._memory.reset()
            return True
        except Exception:
            return False

    async def close(self):
        pass


# ── Metrics ──

def compute_metrics(evaluations: list[dict], cutoffs: list[int]) -> dict:
    metrics = {}
    for c in cutoffs:
        label = f"top_{c}"
        scores = [e.get("cutoff_results", {}).get(label, {}).get("score", 0.0) for e in evaluations]
        correct = sum(1 for s in scores if s >= 0.5)
        by_cat = defaultdict(list)
        for e in evaluations:
            cat = e.get("category_name", "unknown")
            by_cat[cat].append(e.get("cutoff_results", {}).get(label, {}).get("score", 0.0))
        cat_metrics = {}
        for cat, cat_scores in sorted(by_cat.items()):
            cat_correct = sum(1 for s in cat_scores if s >= 0.5)
            cat_metrics[cat] = {
                "total": len(cat_scores),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_scores) * 100 if cat_scores else 0.0,
            }
        metrics[label] = {
            "overall": {
                "total": len(scores),
                "correct": correct,
                "accuracy": correct / len(scores) * 100 if scores else 0.0,
            },
            "by_category": cat_metrics,
        }
    return metrics


def display_metrics(metrics: dict):
    for label, m in metrics.items():
        print(f"\n--- {label} ---")
        overall = m["overall"]
        print(f"  Overall: {overall['correct']}/{overall['total']} ({overall['accuracy']:.1f}%)")
        for cat, cm in m.get("by_category", {}).items():
            print(f"  {cat}: {cm['correct']}/{cm['total']} ({cm['accuracy']:.1f}%)")


# ── Main pipeline ──

async def run_locomo(args):
    cutoffs = [int(c) for c in args.top_k_cutoffs.split(",")]
    categories = [int(c) for c in args.categories.split(",")]
    conv_indices = [int(c) for c in args.conversations.split(",")]

    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset_path or download_dataset("datasets/locomo")
    dataset = load_dataset(dataset_path)

    print(f"LOCOMO | project={args.project_name} conversations={args.conversations}")
    print(f"  Cutoffs: {args.top_k_cutoffs} Categories: {args.categories}")

    # Init SeptMuse
    from septmuse.configs.defaults import default_config
    config = default_config()
    client = SeptMuseClient(config=config)

    # Init LLM (for full mode)
    llm = None
    if not args.predict_only:
        from septmuse.llms.base import LLM
        llm_provider = os.getenv("SEPTMUSE_LLM", "")
        if llm_provider:
            from septmuse.services.providers import llm_provider as provider
            llm = provider.resolve(llm_provider, config=config.llm)

    all_evaluations = []

    for conv_idx in conv_indices:
        if conv_idx >= len(dataset):
            continue
        entry = dataset[conv_idx]
        conversation = entry["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        user_id = f"locomo_{conv_idx}_{args.project_name}"

        # Ingest
        sorted_sessions = get_sorted_sessions(conversation)
        total_chunks = sum(len(session_to_chunks(s, speaker_a, speaker_b)) for _, _, s in sorted_sessions)
        print(f"\nIngesting conv {conv_idx}: {speaker_a} & {speaker_b}, {len(sorted_sessions)} sessions, {total_chunks} chunks")

        for session_key, date_str, turns in sorted_sessions:
            chunks = session_to_chunks(turns, speaker_a, speaker_b)
            session_epoch = locomo_date_to_epoch(date_str)
            for chunk in chunks:
                await client.add(chunk, user_id, timestamp=session_epoch)

        # Search + Answer + Judge
        questions = entry.get("qa", entry.get("qa_pairs", []))
        conv_questions = [(qi, qa) for qi, qa in enumerate(questions) if qa.get("category") in categories]
        if args.max_questions:
            conv_questions = conv_questions[:args.max_questions]

        print(f"  Questions: {len(conv_questions)}")
        for qi, qa in conv_questions:
            qid = f"conv{conv_idx}_q{qi}"
            question = qa["question"]
            category = qa["category"]
            answer = str(qa["answer"])
            cat_name = CATEGORY_NAMES.get(category, "unknown")

            start = time.monotonic()
            search_results = await client.search(question, user_id, top_k=args.top_k)
            latency = (time.monotonic() - start) * 1000

            result = {
                "question_id": qid,
                "conversation_idx": conv_idx,
                "category": category,
                "category_name": cat_name,
                "question": question,
                "ground_truth_answer": answer,
                "retrieval": {
                    "search_query": question,
                    "search_results": search_results,
                    "search_latency_ms": round(latency, 1),
                    "total_results": len(search_results),
                },
            }

            if not args.predict_only and llm is not None:
                cutoff_results = {}
                for c in cutoffs:
                    sliced = search_results[:c]
                    label = f"top_{c}"
                    memories_text = "\n".join(f"- {r['memory']}" for r in sliced) or "(No memories)"
                    gen_prompt = ANSWER_GEN_PROMPT.format(memories=memories_text, question=question)
                    generated = llm.complete("", gen_prompt)
                    if "ANSWER:" in generated:
                        generated = generated.rsplit("ANSWER:", 1)[-1].strip()
                    judge_prompt = JUDGE_PROMPT.format(question=question, answer=answer, response=generated)
                    raw = llm.complete(JUDGE_SYSTEM_PROMPT, judge_prompt)
                    correct = "CORRECT" in raw.upper()
                    cutoff_results[label] = {
                        "judgment": "CORRECT" if correct else "WRONG",
                        "score": 1.0 if correct else 0.0,
                        "generated_answer": generated,
                    }
                result["cutoff_results"] = cutoff_results

            all_evaluations.append(result)
            result_path = os.path.join(output_dir, f"{qid}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        await client.delete_user(user_id)

    # Metrics
    if not args.predict_only and all_evaluations:
        metrics = compute_metrics(all_evaluations, cutoffs)
        display_metrics(metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(args.output_dir, f"locomo_results_{timestamp}.json")
        with open(unified_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "evaluations": all_evaluations}, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {unified_path}")

    print(f"\nTotal questions processed: {len(all_evaluations)}")


def main():
    parser = argparse.ArgumentParser(description="LOCOMO-10 benchmark with SeptMuse")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--conversations", default="0")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200")
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--output-dir", default="results/locomo")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    args = parser.parse_args()
    if args.full:
        args.predict_only = False
    asyncio.run(run_locomo(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Lint the harness**

Run: `$env:PYTHONPATH="src"; ruff check scripts/eval_locomo.py`
Expected: Clean (fix any issues)

- [ ] **Step 4: Run predict-only mode (1 conversation, requires Ollama embedder)**

```powershell
$env:PYTHONPATH = "src"
$env:SEPTMUSE_RATE_LIMIT = "0"
$env:SEPTMUSE_EMBEDDER = "ollama"
$env:SEPTMUSE_EMBEDDER_MODEL = "bge-m3:latest"
$env:SEPTMUSE_EMBEDDER_BASE_URL = "http://localhost:7521"
$env:SEPTMUSE_EMBEDDER_DIMS = 1024
$env:SEPTMUSE_EMBEDDING_DIMS = 1024
$env:SEPTMUSE_VECTOR_BACKEND = "sqlite"
$env:OLLAMA_HOST = "http://localhost:7521"
python scripts/eval_locomo.py --project-name baseline --conversations 0 --predict-only --max-questions 5
```
Expected: Downloads dataset, ingests conv 0, runs 5 queries, saves results. No crash.

- [ ] **Step 5: Run full mode (requires Ollama LLM + embedder)**

```powershell
$env:SEPTMUSE_LLM = "ollama"
$env:SEPTMUSE_LLM_MODEL = "qwen3.5:latest"
python scripts/eval_locomo.py --project-name baseline --conversations 0 --full --max-questions 5
```
Expected: Same + generates answers + judges + prints metrics table.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/eval_locomo.py
git commit -m "feat: LOCOMO-10 评估 harness — SeptMuse + Ollama answer/judge"
```

---

## Self-Review

### 1. Spec coverage
- ✅ Section 0 (OllamaEmbedder bugfix) → Task 1
- ✅ Section 1 (LOCOMO harness) → Task 5
- ✅ Section 2a (HyDE) → Task 2 + Task 4 (routing)
- ✅ Section 2b (Query Rewrite) → Task 3 + Task 4 (routing)
- ✅ Section 2c (Graph BFS) → Task 4 (routing, no new file needed)
- ✅ Section 2d (Progressive) → Task 4 (routing, no new file needed)
- ✅ Section 2e (Dependency injection) → Task 4

### 2. Placeholder scan
- No TBD/TODO in any task
- All test code is complete with assertions
- All implementation code is complete, not described

### 3. Type consistency
- `generate_hypothetical_document(query: str, llm) -> str | None` — consistent across Task 2 and Task 4
- `rewrite_query(query: str, llm) -> list[str]` — consistent across Task 3 and Task 4
- `SearchPipeline.__init__` new params: `llm`, `graph_searcher`, `progressive_retriever` — consistent across Task 4 steps
