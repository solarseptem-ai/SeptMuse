# Smart Memory Butler — Phase 1: MCP 暴露编排 API

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ExperimentalMemory 的 remember/recall/improve/forget 四个编排方法暴露为 MCP 工具，让 agent 通过 MCP 协议直接使用高层记忆管家 API。

**Architecture:** 在现有 `api/mcp/tools.py` 中新增 4 个 `@mcp.tool` 工具，每个工具是薄层委托——参数校验 + 调用 `get_memory_safe()` 拿到 ExperimentalMemory 实例 + 转发到对应 V2 方法 + JSON 序列化返回。不改变现有 18 个底层工具。

**Tech Stack:** Python 3.10+, FastMCP `@mcp.tool` 装饰器, ExperimentalMemory V2 API

## Global Constraints

- **不使用 `from __future__ import annotations`**（FastMCP `func_metadata` 把返回注解当字符串解析会炸，文件顶部有显式注释）
- **工具签名必须用具体类型**（`list[str]` 而非 `list`，`str` 而非 `str | None` 用默认值 `""` 代替 Optional）
- **`user_id` 默认从 contextvars 读**（`_resolve_user_id` 已有函数），错误时返回字符串 `"Error: ..."` 而非异常
- **返回值用 `json.dumps(result, ensure_ascii=False, default=str)`** 序列化
- **测试环境**：`PYTHONPATH=src`, `SEPTMUSE_EMBEDDER=hash`, `SEPTMUSE_TOKENIZER=space`
- **line-length 120**，ruff check + format 必须通过
- **git commit 用** `git -c core.hooksPath=<empty> commit -m "..."` 绕过 LFS hook 挂起

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `src/septmuse/api/mcp/tools.py` | MCP 工具注册 | 修改：新增 4 个 `@mcp.tool` |
| `tests/unit/test_mcp_butler_tools.py` | Phase 1 工具测试 | 新建 |

## 现有代码参考

### MCP 工具注册模式（tools.py 现有范例）

```python
@mcp.tool(description="...")
async def tool_name(param: str, user_id: str = "") -> str:
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided ..."
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.some_method(param, user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error: {e}"
```

### V2 编排 API 签名（ExperimentalMemory / Memory 类）

```python
def remember(self, text: str, *, user_id: str, session_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
    # 返回 {"captured": bool, "raw_id": str|None, "fact_ids": list, "memory_ids": list, "memory_id": str|None}

def recall(self, query: str, *, user_id: str, session_id: str | None = None, top_k: int | None = None, ...) -> dict[str, Any]:
    # 返回 {"memories": list, "injected_prompt": str, "route": str|None, "strategy": str|None, "used_tokens": int}

def improve(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
    # 返回 {"dream": {...}, "rules": int, "coverage": {...}, "conflicts": {...}}

def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
    # 返回 {"event": "FORGET", "invalidated_at": str|None, "deleted_at": str|None}
```

### get_memory_safe() 返回 ExperimentalMemory 实例

```python
from septmuse.api.mcp.server import get_memory_safe, mcp
mem = get_memory_safe()  # → ExperimentalMemory | None
```

---

### Task 1: remember MCP 工具

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`（在文件末尾 `logger.info` 行之前插入）
- Test: `tests/unit/test_mcp_butler_tools.py`

**Interfaces:**
- Consumes: `get_memory_safe()` → `ExperimentalMemory`, `_resolve_user_id()` → `str`, `ExperimentalMemory.remember(text, *, user_id, session_id, agent_id)` → `dict`
- Produces: MCP tool `remember(text: str, user_id: str, agent_id: str, session_id: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for MCP butler tools (remember/recall/improve/forget)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_memory():
    """Mock ExperimentalMemory with V2 methods."""
    m = MagicMock()
    m.remember.return_value = {
        "captured": True,
        "raw_id": "raw-001",
        "fact_ids": ["fact-001", "fact-002"],
        "memory_ids": ["mem-001"],
        "memory_id": "mem-001",
    }
    m.recall.return_value = {
        "memories": [{"id": "mem-001", "memory": "hello", "score": 0.9}],
        "injected_prompt": "<memory>hello</memory>",
        "route": "semantic",
        "strategy": "MAINTAIN",
        "used_tokens": 50,
    }
    m.improve.return_value = {
        "dream": {"links_created": 3, "processed": 5},
        "rules": 2,
        "coverage": {"overall_score": 0.7},
        "conflicts": {"resolved": 1},
    }
    m.forget.return_value = {
        "event": "FORGET",
        "invalidated_at": "2026-08-13T00:00:00Z",
        "deleted_at": "2026-08-13T00:00:01Z",
    }
    return m


def test_remember_tool_basic(mock_memory):
    """remember 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import remember
        import asyncio

        result = asyncio.run(remember("用户要出差去上海", user_id="alice"))
        parsed = json.loads(result)
        assert parsed["captured"] is True
        assert parsed["memory_id"] == "mem-001"
        assert len(parsed["fact_ids"]) == 2
        mock_memory.remember.assert_called_once_with("用户要出差去上海", user_id="alice", session_id=None, agent_id=None)


def test_remember_tool_no_user_id():
    """remember 工具：缺 user_id 返回 Error 字符串。"""
    from septmuse.api.mcp.tools import remember
    import asyncio

    result = asyncio.run(remember("test", user_id=""))
    assert result.startswith("Error:")


def test_remember_tool_no_memory():
    """remember 工具：Memory 不可用返回 Error。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=None):
        from septmuse.api.mcp.tools import remember
        import asyncio

        result = asyncio.run(remember("test", user_id="alice"))
        assert result.startswith("Error:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_remember_tool_basic -v`
Expected: FAIL with `ImportError: cannot import name 'remember' from 'septmuse.api.mcp.tools'`

- [ ] **Step 3: Write minimal implementation**

在 `tools.py` 的 `logger.info("mcp_tools_registered", count=18)` 行**之前**插入：

```python
# ---------------------------------------------------------------------------
# V2 编排 API 工具 (记忆管家 — remember/recall/improve/forget)
# ---------------------------------------------------------------------------


@mcp.tool(
    description="记忆管家：自动捕获文本→抽取事实→分类存储→更新工作记忆。agent只需传入对话文本，SeptMuse自主决定存什么、怎么分类。每次用户透露个人信息、偏好、计划或重要事实时调用。"
)
async def remember(text: str, user_id: str = "", agent_id: str = "", session_id: str = "") -> str:
    """记忆管家写入入口。

    Args:
        text: 要记忆的文本（对话片段、用户陈述等）
        user_id: 用户 ID（空则从 contextvar 读取）
        agent_id: agent ID（可选）
        session_id: 会话 ID（可选，用于会话隔离）

    Returns:
        JSON: {captured, raw_id, fact_ids, memory_ids, memory_id}
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided (传 user_id 参数或设 SEPTMUSE_USER_ID 环境变量)"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        result = mem.remember(
            text,
            user_id=uid,
            session_id=session_id or None,
            agent_id=agent_id or None,
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in remember: {e}"
```

同时更新末尾 count：`count=18` → `count=19`

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_remember_tool_basic tests/unit/test_mcp_butler_tools.py::test_remember_tool_no_user_id tests/unit/test_mcp_butler_tools.py::test_remember_tool_no_memory -v`
Expected: 3 PASS

- [ ] **Step 5: ruff check**

Run: `ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py
git -c core.hooksPath= commit -m "feat(mcp): add remember tool — butler write entry point"
```

---

### Task 2: recall MCP 工具

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_mcp_butler_tools.py`

**Interfaces:**
- Consumes: `ExperimentalMemory.recall(query, *, user_id, session_id, top_k, ...)` → `dict`
- Produces: MCP tool `recall(query: str, user_id: str, agent_id: str, session_id: str, top_k: int, token_budget: int) -> str`

- [ ] **Step 1: Write the failing test**

追加到 `tests/unit/test_mcp_butler_tools.py`：

```python
def test_recall_tool_basic(mock_memory):
    """recall 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import recall
        import asyncio

        result = asyncio.run(recall("上海出差", user_id="alice"))
        parsed = json.loads(result)
        assert "memories" in parsed
        assert "injected_prompt" in parsed
        assert len(parsed["memories"]) == 1
        mock_memory.recall.assert_called_once()


def test_recall_tool_with_top_k(mock_memory):
    """recall 工具：top_k 参数传递。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import recall
        import asyncio

        result = asyncio.run(recall("query", user_id="alice", top_k=10))
        parsed = json.loads(result)
        assert "memories" in parsed
        _, kwargs = mock_memory.recall.call_args
        assert kwargs.get("top_k") == 10 or mock_memory.recall.call_args[0][0] == "query"


def test_recall_tool_no_user_id():
    """recall 工具：缺 user_id 返回 Error。"""
    from septmuse.api.mcp.tools import recall
    import asyncio

    result = asyncio.run(recall("test", user_id=""))
    assert result.startswith("Error:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_recall_tool_basic -v`
Expected: FAIL with `ImportError: cannot import name 'recall'`

- [ ] **Step 3: Write minimal implementation**

在 `remember` 工具之后、`logger.info` 之前插入：

```python
@mcp.tool(
    description="记忆管家：智能检索记忆→压缩摘要→token裁剪→注入prompt→策略推荐。返回即用上下文包（摘要+记忆列表+冲突标注+行动建议）。agent需要回忆相关信息时调用，取代简单search。"
)
async def recall(
    query: str,
    user_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    top_k: int = 5,
    token_budget: int = 2000,
) -> str:
    """记忆管家召回入口。

    Args:
        query: 查询文本
        user_id: 用户 ID（空则从 contextvar 读取）
        agent_id: agent ID（可选）
        session_id: 会话 ID（可选）
        top_k: 返回记忆数上限（默认 5）
        token_budget: token 预算上限（默认 2000）

    Returns:
        JSON: {memories, injected_prompt, route, strategy, used_tokens}
        Phase 2 将新增: summary, conflicts
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided (传 user_id 参数或设 SEPTMUSE_USER_ID 环境变量)"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        result = mem.recall(
            query,
            user_id=uid,
            session_id=session_id or None,
            top_k=top_k,
            inject_prompt=True,
            token_budget=token_budget,
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in recall: {e}"
```

更新末尾 count：`count=19` → `count=20`

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_recall_tool_basic tests/unit/test_mcp_butler_tools.py::test_recall_tool_with_top_k tests/unit/test_mcp_butler_tools.py::test_recall_tool_no_user_id -v`
Expected: 3 PASS

- [ ] **Step 5: ruff check**

Run: `ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py
git -c core.hooksPath= commit -m "feat(mcp): add recall tool — butler read entry point"
```

---

### Task 3: improve MCP 工具

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_mcp_butler_tools.py`

**Interfaces:**
- Consumes: `ExperimentalMemory.improve(*, user_id, limit)` → `dict`
- Produces: MCP tool `improve(user_id: str, tasks: str) -> str`（tasks 是逗号分隔字符串，因为 MCP 工具参数不支持 list[str]）

- [ ] **Step 1: Write the failing test**

追加到 `tests/unit/test_mcp_butler_tools.py`：

```python
def test_improve_tool_basic(mock_memory):
    """improve 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import improve
        import asyncio

        result = asyncio.run(improve(user_id="alice"))
        parsed = json.loads(result)
        assert parsed["dream"]["links_created"] == 3
        assert parsed["rules"] == 2
        mock_memory.improve.assert_called_once_with(user_id="alice", limit=20)


def test_improve_tool_no_user_id():
    """improve 工具：缺 user_id 返回 Error。"""
    from septmuse.api.mcp.tools import improve
    import asyncio

    result = asyncio.run(improve(user_id=""))
    assert result.startswith("Error:")


def test_improve_tool_no_memory():
    """improve 工具：Memory 不可用返回 Error。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=None):
        from septmuse.api.mcp.tools import improve
        import asyncio

        result = asyncio.run(improve(user_id="alice"))
        assert result.startswith("Error:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_improve_tool_basic -v`
Expected: FAIL with `ImportError: cannot import name 'improve'`

- [ ] **Step 3: Write minimal implementation**

在 `recall` 工具之后、`logger.info` 之前插入：

```python
@mcp.tool(
    description="记忆管家：整理记忆→蒸馏教训→解决冲突→图谱生长→覆盖分析。在对话结束或空闲时调用，让SeptMuse整理和强化记忆。tasks可选指定子任务(distill/resolve_conflicts/grow_graph/analyze_coverage)，不传跑全部。"
)
async def improve(user_id: str = "", tasks: str = "") -> str:
    """记忆管家演化入口。

    Args:
        user_id: 用户 ID（空则从 contextvar 读取）
        tasks: 指定子任务，逗号分隔（如 "distill,resolve_conflicts"）。空=全部

    Returns:
        JSON: {dream, rules, coverage, conflicts}
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided (传 user_id 参数或设 SEPTMUSE_USER_ID 环境变量)"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        result = mem.improve(user_id=uid)
        # Phase 4: 按 tasks 过滤子任务
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in improve: {e}"
```

更新末尾 count：`count=20` → `count=21`

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_improve_tool_basic tests/unit/test_mcp_butler_tools.py::test_improve_tool_no_user_id tests/unit/test_mcp_butler_tools.py::test_improve_tool_no_memory -v`
Expected: 3 PASS

- [ ] **Step 5: ruff check**

Run: `ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py
git -c core.hooksPath= commit -m "feat(mcp): add improve tool — butler evolution entry point"
```

---

### Task 4: forget MCP 工具

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_mcp_butler_tools.py`

**Interfaces:**
- Consumes: `ExperimentalMemory.forget(memory_id, *, user_id)` → `dict`
- Produces: MCP tool `forget(user_id: str, memory_id: str) -> str`

- [ ] **Step 1: Write the failing test**

追加到 `tests/unit/test_mcp_butler_tools.py`：

```python
def test_forget_tool_by_memory_id(mock_memory):
    """forget 工具：按 memory_id 遗忘。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import forget
        import asyncio

        result = asyncio.run(forget(user_id="alice", memory_id="mem-001"))
        parsed = json.loads(result)
        assert parsed["event"] == "FORGET"
        assert parsed["invalidated_at"] is not None
        mock_memory.forget.assert_called_once_with("mem-001", user_id="alice")


def test_forget_tool_no_user_id():
    """forget 工具：缺 user_id 返回 Error。"""
    from septmuse.api.mcp.tools import forget
    import asyncio

    result = asyncio.run(forget(user_id="", memory_id="mem-001"))
    assert result.startswith("Error:")


def test_forget_tool_no_memory_id(mock_memory):
    """forget 工具：缺 memory_id 返回 Error。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import forget
        import asyncio

        result = asyncio.run(forget(user_id="alice", memory_id=""))
        assert result.startswith("Error:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_forget_tool_by_memory_id -v`
Expected: FAIL with `ImportError: cannot import name 'forget'`

- [ ] **Step 3: Write minimal implementation**

在 `improve` 工具之后、`logger.info` 之前插入：

```python
@mcp.tool(
    description="记忆管家：遗忘指定记忆（先invalidate标记失效，再软删除）。memory_id=要遗忘的记忆ID。用于清理过时信息或用户要求忘记的内容。"
)
async def forget(user_id: str = "", memory_id: str = "") -> str:
    """记忆管家遗忘入口。

    Args:
        user_id: 用户 ID（空则从 contextvar 读取）
        memory_id: 要遗忘的记忆 ID（必填）

    Returns:
        JSON: {event, invalidated_at, deleted_at}
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided (传 user_id 参数或设 SEPTMUSE_USER_ID 环境变量)"

    if not memory_id:
        return "Error: memory_id is required"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        result = mem.forget(memory_id, user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in forget: {e}"
```

更新末尾 count：`count=21` → `count=22`

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_forget_tool_by_memory_id tests/unit/test_mcp_butler_tools.py::test_forget_tool_no_user_id tests/unit/test_mcp_butler_tools.py::test_forget_tool_no_memory_id -v`
Expected: 3 PASS

- [ ] **Step 5: ruff check**

Run: `ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py
git -c core.hooksPath= commit -m "feat(mcp): add forget tool — butler forget entry point"
```

---

### Task 5: 全量回归测试

**Files:**
- 无新建/修改

- [ ] **Step 1: 运行全部 MCP butler 工具测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py -v`
Expected: 12 PASS

- [ ] **Step 2: 运行现有 MCP 工具测试（无回归）**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_tools.py -v`
Expected: 全部 PASS（无 regression）

- [ ] **Step 3: 运行 health + auth 测试（确认中间件链无破坏）**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_health.py tests/unit/test_auth.py -v`
Expected: 全部 PASS

- [ ] **Step 4: 运行 e2e 测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/e2e/ -q`
Expected: 31 passed, 1 failed (pre-existing test_capture_persists)

- [ ] **Step 5: ruff format check**

Run: `ruff format --check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Expected: All checks passed

- [ ] **Step 6: 最终 commit（如有 format 修复）**

```bash
git add -A
git -c core.hooksPath= commit -m "test(mcp): full regression for butler tools — 12 pass, no regression"
```
