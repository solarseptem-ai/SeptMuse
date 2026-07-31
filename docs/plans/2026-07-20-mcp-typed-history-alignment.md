# MCP 工具对齐 + 类型化 update + history API 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 从高到低补齐 4 个未对齐能力：MCP 4 stub 接真实实现 + MCP 新增 6 工具 + 类型化记忆 update + history API。

**Architecture:** 全部是 facade 已有能力的暴露层补齐。MCP tools.py 接上 facade 真实方法；TypedMemoryStore 加 update_fact/episode/rule；store 加 get_history；REST+CLI 暴露 history。

**Tech Stack:** Python 3.10+, FastMCP, SQLModel, FastAPI, argparse

## Global Constraints

- ruff line-length 120，规则 `E F I W UP B SIM RUF`（忽略 E501 RUF002 RUF003）
- PYTHONPATH=src 运行 pytest
- 项目非 git repo，无 commit 步骤
- MCP tools.py 不用 `from __future__ import annotations`（FastMCP func_metadata 限制）
- MCP 工具参数名不用 `result`（与 FastMCP 返回值 `result` 字段冲突）
- 全回归基线：537 passed, 9 skipped, 1 deselected

---

### Task 1: MCP 4 个 stub 接真实实现

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_mcp_tools.py`

**Interfaces:**
- Consumes: `Memory.add_episode`/`counterfactual`/`rehearse`/`find_rehearse_candidates`/`coverage_report`

- [ ] **Step 1: 写 4 stub 接真实实现的测试**

在 `tests/unit/test_mcp_tools.py` 追加（需先读现有测试模式确认 import）：

```python
class TestMcpStubReplaced:
    @pytest.mark.asyncio
    async def test_remember_episode_real(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.remember_episode(
            observation="saw error", thoughts="try X", action="did X", outcome="worked", user_id="alice"
        )
        assert "Episode recorded" not in result  # 不再是 stub 字符串
        assert "id" in result or "episode" in result.lower()

    @pytest.mark.asyncio
    async def test_causal_query_real(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.causal_query(cause_event_id="evt-1", hypothesized_effect="evt-2", user_id="alice")
        assert "阶段4" not in result  # 不再是 stub

    @pytest.mark.asyncio
    async def test_rehearse_real(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.rehearse(user_id="alice")
        assert "阶段4" not in result

    @pytest.mark.asyncio
    async def test_coverage_report_real(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.coverage_report(user_id="alice")
        assert "阶段4" not in result
```

- [ ] **Step 2: 运行验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_mcp_tools.py::TestMcpStubReplaced -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（返回 stub 字符串）

- [ ] **Step 3: 接上真实实现**

替换 `tools.py` 中 4 个 stub 函数体：

```python
@mcp.tool(description="记录成功交互的推理经验 (观察/思考/行动/结果)。借鉴 LangMem Episode。")
async def remember_episode(observation: str, thoughts: str, action: str, outcome: str, user_id: str = ""):
    """记录推理经验 (架构文档 §3.2.1)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.add_episode(
            f"obs: {observation}; act: {action}; result: {outcome}",
            user_id=uid,
            event_type="reasoning",
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=outcome,
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error recording episode: {e}"


@mcp.tool(description="反事实因果查询: 若某事件未发生,结果是否仍成立 (架构文档 §6.1)")
async def causal_query(cause_event_id: str, hypothesized_effect: str, user_id: str = ""):
    """因果查询 (架构文档 §6.1)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.counterfactual(cause_event_id, hypothesized_effect, user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in causal query: {e}"


@mcp.tool(description="触发主动复述强化低强度高价值记忆 (架构文档 §6.2)")
async def rehearse(user_id: str = "", memory_id: str = ""):
    """主动复述 (架构文档 §6.2)。memory_id 为空时批量复述候选。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        if memory_id:
            result = mem.rehearse(memory_id, user_id=uid)
            return json.dumps(result, ensure_ascii=False, default=str)
        candidates = mem.find_rehearse_candidates(user_id=uid)
        count = 0
        for c in candidates:
            mem.rehearse(c["memory_id"], user_id=uid)
            count += 1
        return json.dumps({"rehearsed": count, "candidates": len(candidates)}, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in rehearse: {e}"


@mcp.tool(description="生成元认知覆盖报告: agent 记住了什么/记不住什么 (架构文档 §6.3)")
async def coverage_report(user_id: str = ""):
    """覆盖报告 (架构文档 §6.3)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.coverage_report(user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error generating coverage report: {e}"
```

- [ ] **Step 4: 运行验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_mcp_tools.py::TestMcpStubReplaced -v 2>&1 | Select-Object -Last 10`
Expected: 4 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_tools.py`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 541 passed, 9 skipped, 1 deselected

---

### Task 2: MCP 新增 6 工具 (update/block/history)

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_mcp_tools.py`

**Interfaces:**
- Consumes: `Memory.update`/`update_block`/`core_memory_append`/`core_memory_replace`/`get_blocks`/`get_history`(Task 4)

- [ ] **Step 1: 写 6 新工具的测试**

在 `tests/unit/test_mcp_tools.py` 追加：

```python
class TestMcpNewTools:
    @pytest.mark.asyncio
    async def test_update_memory(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        add_result = mem.add("old", user_id="alice")
        mid = add_result["results"][0]["id"]
        result = await tools.update_memory(memory_id=mid, text="new text", user_id="alice")
        assert "UPDATE" in result

    @pytest.mark.asyncio
    async def test_update_block(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.update_block(agent_id="ag1", label="human", value="Alice", user_id="alice")
        assert "UPDATE" in result

    @pytest.mark.asyncio
    async def test_core_memory_append_tool(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        mem.update_block("ag1", "human", "base")
        result = await tools.core_memory_append(agent_id="ag1", label="human", content="appended", user_id="alice")
        assert "APPEND" in result

    @pytest.mark.asyncio
    async def test_core_memory_replace_tool(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        mem.update_block("ag1", "human", "old text")
        result = await tools.core_memory_replace(agent_id="ag1", label="human", old_content="old", new_content="new", user_id="alice")
        assert "REPLACE" in result

    @pytest.mark.asyncio
    async def test_get_blocks_tool(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        result = await tools.get_blocks(agent_id="ag1", user_id="alice")
        assert "human" in result

    @pytest.mark.asyncio
    async def test_get_history_tool(self, tmp_path, monkeypatch):
        from septmuse.api.mcp import tools
        from septmuse import Memory, MemoryConfig
        from septmuse.providers.embedders.hash import HashEmbedder
        mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "t.db")), embedder=HashEmbedder())
        monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
        add_result = mem.add("content", user_id="alice")
        mid = add_result["results"][0]["id"]
        result = await tools.get_memory_history(memory_id=mid, user_id="alice")
        assert "ADD" in result
```

- [ ] **Step 2: 运行验证失败**

Expected: FAIL（`update_memory` 等函数不存在 → AttributeError）

- [ ] **Step 3: 加 6 个新工具**

在 `tools.py` 末尾（`coverage_report` 之后）加：

```python
# ---------------------------------------------------------------------------
# SeptMuse 扩展 6 工具 (update + block + history, 对齐 mem0 plugin 9 工具)
# ---------------------------------------------------------------------------


@mcp.tool(description="更新已有记忆的内容或 metadata。")
async def update_memory(memory_id: str, text: str = "", user_id: str = "", metadata: str = ""):
    """更新记忆 (对齐 mem0 update_memory)。metadata 传 JSON 字符串或空。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        meta = json.loads(metadata) if metadata else None
        result = mem.update(memory_id, text or None, user_id=uid, metadata=meta)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error updating memory: {e}"


@mcp.tool(description="更新工作记忆 Block 的值 (对齐 Letta update_block_value)。")
async def update_block(agent_id: str, label: str, value: str, user_id: str = ""):
    """更新 block value。"""
    uid = _resolve_user_id(user_id)
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.update_block(agent_id, label, value)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error updating block: {e}"


@mcp.tool(description="追加内容到工作记忆 Block (对齐 Letta core_memory_append)。")
async def core_memory_append(agent_id: str, label: str, content: str, user_id: str = ""):
    """追加 block 内容。"""
    uid = _resolve_user_id(user_id)
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.core_memory_append(agent_id, label, content)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error appending to block: {e}"


@mcp.tool(description="替换工作记忆 Block 中的内容片段 (对齐 Letta core_memory_replace)。")
async def core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str, user_id: str = ""):
    """替换 block 内容片段。"""
    uid = _resolve_user_id(user_id)
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.core_memory_replace(agent_id, label, old_content, new_content)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error replacing block content: {e}"


@mcp.tool(description="列出 agent 的工作记忆 Block 列表。")
async def get_blocks(agent_id: str, user_id: str = ""):
    """列出 block。"""
    uid = _resolve_user_id(user_id)
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        blocks = mem.get_blocks(agent_id)
        return json.dumps(blocks, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error getting blocks: {e}"


@mcp.tool(description="查看记忆的变更历史 (ADD/UPDATE/DELETE 记录)。")
async def get_memory_history(memory_id: str, user_id: str = ""):
    """查看记忆历史。"""
    uid = _resolve_user_id(user_id)
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        history = mem.get_history(memory_id)
        return json.dumps(history, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error getting history: {e}"
```

注意：`get_memory_history` 依赖 Task 4 的 `Memory.get_history`。如果 Task 4 还没实现，这个工具会报错。所以 Task 4 必须在 Task 2 之前或同时实现。**调整执行顺序：先 Task 4（history API），再 Task 2。**

- [ ] **Step 4: 运行验证通过**（Task 4 完成后）

- [ ] **Step 5: ruff + 全回归**

Expected: 547 passed（541 + 6 新增）

---

### Task 3: 类型化记忆 update (TypedMemoryStore)

**Files:**
- Modify: `src/septmuse/storage/typed_store.py`
- Modify: `src/septmuse/orchestration/memory.py`（facade 包装）
- Test: `tests/unit/test_update.py`

**Interfaces:**
- Produces: `TypedMemoryStore.update_fact`/`update_episode`/`update_rule` + `Memory.update_fact`/`update_episode`/`update_rule`

- [ ] **Step 1: 写类型化 update 测试**

在 `tests/unit/test_update.py` 追加：

```python
class TestTypedUpdate:
    def test_update_fact(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_fact("alice", "likes", "python", user_id="alice")
        fid = result["id"]
        updated = m.update_fact(fid, subject="alice", predicate="likes", object="rust", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_episode_content(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_episode("old event", user_id="alice")
        eid = result["id"]
        updated = m.update_episode(eid, content="new event", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_rule(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add_rule("old rule", user_id="alice")
        rid = result["id"]
        updated = m.update_rule(rid, rule="new rule", user_id="alice")
        assert updated["event"] == "UPDATE"

    def test_update_fact_not_found(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update_fact("nonexistent", subject="x", predicate="y", object="z", user_id="alice")
        assert result["event"] == "NOT_FOUND"
```

- [ ] **Step 2: 运行验证失败**

Expected: FAIL（`update_fact` 不存在）

- [ ] **Step 3: TypedMemoryStore 加 update 方法**

在 `typed_store.py` 的 TypedMemoryStore 类加（在对应 add 方法之后）：

```python
    def update_fact(
        self, fact_id: str, subject: str, predicate: str, object: str
    ) -> SemanticFact | None:
        """更新语义事实。"""
        with Session(self.engine) as session:
            fact = session.get(SemanticFact, fact_id)
            if not fact or fact.is_deleted:
                return None
            fact.subject = subject
            fact.predicate = predicate
            fact.object = object
            fact.touch()
            session.add(fact)
            session.commit()
            session.refresh(fact)
            return fact

    def update_episode(self, episode_id: str, content: str) -> EpisodicEvent | None:
        """更新情节事件内容。"""
        with Session(self.engine) as session:
            event = session.get(EpisodicEvent, episode_id)
            if not event or event.is_deleted:
                return None
            event.content = content
            event.touch()
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def update_rule(self, rule_id: str, rule: str) -> ProceduralRule | None:
        """更新程序规则。"""
        with Session(self.engine) as session:
            r = session.get(ProceduralRule, rule_id)
            if not r or r.is_deleted:
                return None
            r.rule = rule
            r.touch()
            session.add(r)
            session.commit()
            session.refresh(r)
            return r
```

- [ ] **Step 4: Memory facade 加包装方法**

在 `memory.py` 加（在对应 add 方法之后）：

```python
    def update_fact(
        self, fact_id: str, *, subject: str, predicate: str, object: str, user_id: str
    ) -> dict[str, Any]:
        """更新语义事实。"""
        fact = self.typed_store.update_fact(fact_id, subject, predicate, object)
        if fact is None:
            return {"id": fact_id, "event": "NOT_FOUND"}
        return {"id": fact.id, "triple": [subject, predicate, object], "event": "UPDATE"}

    def update_episode(self, episode_id: str, *, content: str, user_id: str) -> dict[str, Any]:
        """更新情节事件。"""
        event = self.typed_store.update_episode(episode_id, content)
        if event is None:
            return {"id": episode_id, "event": "NOT_FOUND"}
        return {"id": event.id, "content": content, "event": "UPDATE"}

    def update_rule(self, rule_id: str, *, rule: str, user_id: str) -> dict[str, Any]:
        """更新程序规则。"""
        r = self.typed_store.update_rule(rule_id, rule)
        if r is None:
            return {"id": rule_id, "event": "NOT_FOUND"}
        return {"id": r.id, "rule": rule, "event": "UPDATE"}
```

- [ ] **Step 5: 运行验证通过 + ruff + 全回归**

Expected: 4 passed, 551 total（547 + 4）

---

### Task 4: history API (store + facade + REST + CLI)

**Files:**
- Modify: `src/septmuse/storage/base.py`（ABC 加 get_history）
- Modify: `src/septmuse/storage/sqlite/store.py`（实现）
- Modify: `src/septmuse/orchestration/memory.py`（facade）
- Modify: `src/septmuse/api/rest/__init__.py`（REST 端点）
- Modify: `src/septmuse/cli/main.py`（CLI 命令）
- Test: `tests/unit/test_update.py`

**Interfaces:**
- Produces: `MemoryStore.get_history(memory_id) -> list[dict]` + `Memory.get_history(memory_id) -> list[dict]`

- [ ] **Step 1: 写 history 测试**

在 `tests/unit/test_update.py` 追加：

```python
class TestHistory:
    def test_get_history_after_add(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("content", user_id="alice")
        mid = result["results"][0]["id"]
        history = m.get_history(mid)
        assert len(history) >= 1
        assert history[0]["event"] == "ADD"

    def test_get_history_after_update(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("old", user_id="alice")
        mid = result["results"][0]["id"]
        m.update(mid, "new")
        history = m.get_history(mid)
        events = [h["event"] for h in history]
        assert "ADD" in events
        assert "UPDATE" in events

    def test_get_history_empty(self, tmp_path):
        m = _make_memory(tmp_path)
        history = m.get_history("nonexistent")
        assert history == []

    def test_rest_get_history(self, tmp_path):
        from fastapi.testclient import TestClient
        from septmuse.api.rest import create_app
        m = _make_memory(tmp_path)
        app = create_app(m)
        result = m.add("content", user_id="alice")
        mid = result["results"][0]["id"]
        client = TestClient(app)
        resp = client.get(f"/memories/{mid}/history")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_cli_history(self, tmp_path, monkeypatch, capsys):
        import sys
        from septmuse.cli.main import main
        db = str(tmp_path / "t.db")
        monkeypatch.setattr(sys, "argv", ["septmuse", "init", "--user", "alice", "--db-path", db])
        main()
        monkeypatch.setattr(sys, "argv", ["septmuse", "add", "content", "--user", "alice", "--db-path", db])
        main()
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["septmuse", "add", "content2", "--user", "alice", "--db-path", db])
        main()
        out2 = capsys.readouterr().out
        mid = json.loads(out2).get("memory_id")
        monkeypatch.setattr(sys, "argv", ["septmuse", "history", mid, "--db-path", db])
        main()
        out3 = capsys.readouterr().out
        history = json.loads(out3)
        assert len(history) >= 1
```

- [ ] **Step 2: 运行验证失败**

Expected: FAIL（`get_history` 不存在）

- [ ] **Step 3: store 层加 get_history**

在 `base.py` MemoryStore ABC 加（在 `update` 之后）：

```python
    @abstractmethod
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        ...
```

在 `sqlite/store.py` SQLiteMemoryStore 加（在 `update` 之后）：

```python
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        with self._lock:
            cur = self.conn.execute(
                """SELECT id, memory_id, old_memory, new_memory, event, created_at, is_deleted
                   FROM history WHERE memory_id=? ORDER BY created_at""",
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "is_deleted": bool(r[6]),
            }
            for r in rows
        ]
```

在 `pgvector.py` PGVectorStore 加（对齐 SQLite 模式，用 SQL 查询 history 表，或返回空列表如果无 history 表）。简单回退：

```python
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (PGVector 暂不支持 history, 返回空)。"""
        return []
```

- [ ] **Step 4: facade 加 get_history**

在 `memory.py` Memory 类加（在 `update` 之后）：

```python
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (对齐 mem0 history)。"""
        return self.store.get_history(memory_id)
```

- [ ] **Step 5: REST 加 history 端点**

在 `api/rest/__init__.py` 的 `register_routes` 加（在 `update_memory` 之后，注意路由顺序——`/memories/{memory_id}/history` 要在 `/memories/{memory_id}` 之前或用不同路径模式）：

```python
    @app.get("/memories/{memory_id}/history")
    async def get_history(memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (对齐 mem0 GET /memories/{id}/history)。"""
        return app.state.memory.get_history(memory_id)
```

- [ ] **Step 6: CLI 加 history 命令**

在 `cli/main.py` 的 `_build_parser` 加（在 `update` parser 之后）：

```python
    # history
    p_history = sub.add_parser("history", help="查看记忆变更历史")
    p_history.add_argument("memory_id", help="记忆 ID")
    p_history.add_argument("--db-path", default=None, help="SQLite 路径")
    p_history.set_defaults(func=_cmd_history)
```

加实现函数（在 `_cmd_update` 之后）：

```python
def _cmd_history(args: argparse.Namespace) -> int:
    """查看记忆变更历史。"""
    m = _make_memory(args.db_path)
    history = m.get_history(args.memory_id)
    print(json.dumps(history, ensure_ascii=False, default=str, indent=2))
    return 0
```

- [ ] **Step 7: 运行验证通过 + ruff + 全回归**

Expected: 5 passed, 556 total（551 + 5）

---

### Task 5: 最终全回归 + 冒烟

- [ ] **Step 1: ruff 全量**

Run: `$env:PYTHONPATH="src"; ruff check src/ tests/`
Expected: `All checks passed!`

- [ ] **Step 2: 全回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 556 passed, 9 skipped, 1 deselected

- [ ] **Step 3: MCP 工具数验证**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.api.mcp.tools import mcp; print(len(mcp._tool_manager._tools))"`
Expected: 15（基础 5 + 扩展 4 + 新增 6）

- [ ] **Step 4: CLI --help 验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main --help`
Expected: 列出 init/add/search/dump/update/block/history/serve/mcp/version 10 命令

---

## 执行顺序调整

Task 2 的 `get_memory_history` 工具依赖 Task 4 的 `Memory.get_history`。因此执行顺序为：

**Task 4 → Task 1 → Task 3 → Task 2 → Task 5**

## Self-Review

1. Spec coverage: #1 MCP stub → Task 1 ✓, #2 MCP 新工具 → Task 2 ✓, #3 类型化 update → Task 3 ✓, #4 history API → Task 4 ✓
2. 无占位符
3. 类型一致: `Memory.get_history(memory_id) -> list[dict]` Task 4 定义, Task 2 MCP 调用 ✓
