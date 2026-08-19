# Task 1: search() 新增参数 + hyde/query_rewrite no-op

## Files
- Modify: `src/septmuse/memory/main.py:550-565`（search 签名）
- Modify: `src/septmuse/memory/main.py:584`（recipe 处理后，插入 no-op）
- Test: `tests/unit/test_search_enhanced.py`（新增 2 个测试）

## Interfaces
- Produces: `Memory.search()` 新增 5 个 keyword-only 参数: `forgetting: bool = False`, `token_budget: int | None = None`, `inject_prompt: bool = False`, `hyde: bool = False`, `query_rewrite: bool = False`；返回类型 `list[dict] | dict`

## Steps

### Step 1: 写 hyde/query_rewrite no-op 测试

在 `tests/unit/test_search_enhanced.py` 末尾追加:

```python
def test_search_hyde_noop(memory):
    """search(hyde=True) 不报错, 返回与默认一致。"""
    memory.add("test fact", user_id="alice", infer=False)
    results = memory.search("test", user_id="alice", hyde=True)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_query_rewrite_noop(memory):
    """search(query_rewrite=True) 不报错, 返回与默认一致。"""
    memory.add("test fact", user_id="alice", infer=False)
    results = memory.search("test", user_id="alice", query_rewrite=True)
    assert isinstance(results, list)
    assert len(results) > 0
```

### Step 2: 运行测试验证失败

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_hyde_noop tests/unit/test_search_enhanced.py::test_search_query_rewrite_noop -v
```

Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'hyde'`

### Step 3: 修改 search() 签名 + 加 no-op

在 `src/septmuse/memory/main.py` 的 `search()` 方法签名中，在 `show_expired: bool = False,` 之后添加 5 个新参数:

```python
        show_expired: bool = False,
        # B 类高级参数
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
        hyde: bool = False,
        query_rewrite: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
```

在 recipe 处理之后（约 `main.py:591` 之后，`tk = top_k...` 之前）插入 no-op:

```python
        if hyde:
            logger.info("hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
        if query_rewrite:
            logger.info("query_rewrite_not_enabled", hint="requires LLM (future)")
```

### Step 4: 运行测试验证通过

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_hyde_noop tests/unit/test_search_enhanced.py::test_search_query_rewrite_noop -v
```

Expected: PASS

### Step 5: 验证默认行为不变

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_default_returns_list -v
```

Expected: PASS（默认返回 list）

### Step 6: ruff 检查

```
ruff check src/septmuse/memory/main.py tests/unit/test_search_enhanced.py
```

Expected: No errors

### Step 7: 提交

```bash
git add src/septmuse/memory/main.py tests/unit/test_search_enhanced.py
git commit -m "feat(search): add forgetting/token_budget/inject_prompt/hyde/query_rewrite params + no-op for hyde/query_rewrite"
```
