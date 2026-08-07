# 检索质量 + 生命周期 4 子项设计

## 元信息

- **日期**: 2026-08-07
- **状态**: 待审阅
- **目标**: HyDE 假设文档检索 + 上下文感知查询改写 + 遗忘曲线参数化 + 压缩归档 facade
- **spec**: `docs/superpowers/specs/2026-08-07-retrieval-lifecycle-design.md`
- **不提交**: 按用户指示暂不 commit

---

## 1. 背景

记忆可用性改造（7 Task）已完成：add 决策化 + V2 降级委托 + 用户画像聚合。下一步提升检索质量和生命周期管理。

4 个高 ROI 子项：
1. **HyDE** — query 与记忆的语义鸿沟
2. **查询改写** — 多轮对话指代消解
3. **遗忘参数化** — 不同场景不同衰减
4. **压缩归档** — 记忆膨胀后性能不退化

---

## 2. 子项 1：HyDE 假设文档检索

### 现状
`Memory.search` 用 query 原文 embedding 检索。query "我喜欢什么编程语言" 与记忆 "用户喜欢 Python" 的 embedding 有语义鸿沟。

### 方案
`Memory.search` 加 `hyde: bool = False` 参数。hyde=True 时：
1. LLM 基于 query 生成一段假设答案（~2-3 句）
2. 用假设答案 embedding 检索（假设答案更贴近记忆内容表述）
3. 无 LLM 降级原文检索

### 改动
- 新增 `src/septmuse/prompts/hyde.py` — `HYDE_PROMPT`
- 改 `src/septmuse/memory/main.py` `search` — hyde 分支
- 测试 `tests/unit/test_hyde.py`

### HyDE prompt
```
Given the query, generate a hypothetical answer (2-3 sentences) that would be a good match for retrieving relevant memories. Do not answer the question directly — write what the stored memory would look like.

Query: {query}
Hypothetical answer:
```

### 数据流
```
search(query, hyde=True, user_id=)
  if hyde and self.llm is not None:
    hypothetical = self.llm.complete(HYDE_PROMPT, query)  # 假设答案
    emb = self.embedder.embed(hypothetical)                # 用假设 embedding
  else:
    emb = self.embedder.embed(query)                       # 降级原文
  # 后续检索不变 (hybrid/vector 分支用 emb)
```

---

## 3. 子项 2：上下文感知查询改写

### 现状
多轮对话中用户说"他上次说的那个"，query 原文无法检索。

### 方案
`Memory.search` 加 `query_rewrite: bool = False` 参数。query_rewrite=True + session_id 时：
1. 从 episodic store 取近期 N 条消息作为上下文
2. LLM 改写 query（"他上次说的那个"→"张三提到的项目名称"）
3. 用改写后的 query 检索
4. 无 LLM/无 session_id 降级原文

### 改动
- 新增 `src/septmuse/prompts/rewrite.py` — `QUERY_REWRITE_PROMPT`
- 改 `src/septmuse/memory/main.py` `search` — query_rewrite 分支
- 测试 `tests/unit/test_query_rewrite.py`

### 改写 prompt
```
Given the conversation context and the current query, rewrite the query to be self-contained and searchable. If the query is already clear, return it unchanged.

Conversation context:
{recent_messages}

Current query: {query}

Rewritten query:
```

### 数据流
```
search(query, query_rewrite=True, user_id=, session_id=)
  if query_rewrite and self.llm is not None and session_id:
    recent = self.episodic.get_episodes(user_id=user_id, limit=5)  # 近 5 条
    context = "\n".join([e.content for e in recent])
    rewritten = self.llm.complete(QUERY_REWRITE_PROMPT, f"{context}\n\n{query}")
    query = rewritten.strip() or query
  # 后续检索用改写后的 query
```

---

## 4. 子项 3：遗忘曲线参数化

### 现状
`MemoryStrength.decay(now)` 半衰期硬编码在模型里。`ForgettingRetriever` 无法配置。

### 方案
- `MemoryStrength.decay(now, half_life_days=7.0)` — 加参数
- `ForgettingRetriever.__init__(typed_store, half_life_days=7.0)` — 加参数
- `MemoryConfig` 加 `forgetting_half_life_days: float = 7.0`
- `Memory.__init__` 创建 ForgettingRetriever 时传入配置值

### 改动
- 改 `src/septmuse/models/strength.py` — decay 加 half_life_days
- 改 `src/septmuse/retrieval/forgetting.py` — __init__ 加参数 + apply_strength 传参
- 改 `src/septmuse/configs/defaults.py` — MemoryConfig 加字段
- 改 `src/septmuse/memory/main.py` — 创建 ForgettingRetriever 时传配置值
- 测试 `tests/unit/test_forgetting_params.py`

### 衰减公式（不变，参数化半衰期）
```python
# 指数衰减: strength * 0.5^(elapsed_days / half_life_days)
# half_life_days=7.0: 7 天后强度减半
# half_life_days=inf: 永不衰减 (永久记忆)
# half_life_days=1.0: 1 天后减半 (会话级)
```

---

## 5. 子项 4：压缩归档 facade

### 现状
`Summarizer.compress(user_id, mode, buffer_size)` 已存在（`evolution/summarizer.py`），但需手动实例化 Summarizer。

### 方案
- `Memory.__init__` 创建 `self.summarizer = Summarizer(self.store, self.typed_store, self.llm)`
- `Memory.compress(user_id, mode="static", buffer_size=20)` — facade 委托 Summarizer
- `Memory.recall` 加 `auto_compress: bool = False` 参数 — 记忆数 > 阈值时自动压缩
- Summarizer 本身不改（复用现有 compress）

### 改动
- 改 `src/septmuse/memory/main.py` — __init__ 加 Summarizer + compress facade + recall auto_compress
- 测试 `tests/unit/test_compress_facade.py`

### 数据流
```
Memory.compress(user_id, mode="static", buffer_size=20)
  → self.summarizer.compress(user_id=user_id, mode=mode, buffer_size=buffer_size)

Memory.recall(query, auto_compress=True)
  → 正常检索
  → if auto_compress and len(all_memories) > threshold: self.compress(...)
```

---

## 6. 不破坏承诺

- `Memory.search` 签名向后兼容（新参数 hyde/query_rewrite 默认 False）
- `MemoryStrength.decay(now)` 向后兼容（half_life_days 默认 7.0 = 原硬编码值）
- `ForgettingRetriever.__init__` 向后兼容（half_life_days 默认 7.0）
- `MemoryConfig` 新字段有默认值
- `Memory.recall` 向后兼容（auto_compress 默认 False）
- 无 LLM 时全部降级（hyde/query_rewrite 降级原文，compress 降级拼接摘要）
- 现有测试不改断言

---

## 7. 测试策略

| 测试文件 | 覆盖 |
|----------|------|
| `test_hyde.py` | hyde=True 走 LLM 假设 / hyde=False 走原文 / 无 LLM 降级 |
| `test_query_rewrite.py` | query_rewrite=True + session 改写 / 无 session 降级 / 无 LLM 降级 |
| `test_forgetting_params.py` | half_life_days=1.0 快衰减 / =inf 不衰减 / 默认 7.0 兼容 |
| `test_compress_facade.py` | Memory.compress 委托 / recall auto_compress 触发 |
