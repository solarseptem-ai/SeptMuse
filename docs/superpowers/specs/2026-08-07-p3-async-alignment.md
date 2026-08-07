# P3: AsyncMemory 对齐设计

> 日期: 2026-08-07
> 状态: 待审阅

## 1. 背景

P0-P2 完成后, sync Memory facade 已对齐 mem0 V3。AsyncMemory 仍停留在基础 CRUD, 缺:

| # | 差距 | 影响 |
|---|------|------|
| 1 | add 不决策 (恒 ADD) + 无 memory_type 路由 + 无 P1 参数 | REST API 异步路径缺决策/类型化/过期/归因 |
| 2 | search 无 hybrid/reranker/recipe/hyde 等 | 异步检索只有纯向量 |
| 3 | update 不重链接实体 + delete 不 invalidate | 数据不一致 |
| 4 | 无 remember/recall/forget/improve | 缺 V2 编排层 |

## 2. 改造

**核心策略**: AsyncMemory 已有 `self._sync` (ExperimentalMemory 共享同 DB), 高级方法已有 `asyncio.to_thread` 委托模式 (cognify/reflect/compress 等)。复用此模式:

### Task 1: add 增强

- `memory_type` 路由 (fact/episode/rule/procedural) → `to_thread(self._sync.add, ...)`
- `infer=True` 决策 → `to_thread(self._sync.add, ...)` (用 FactExtractor)
- `infer=False` verbatim → 保留当前真 async 路径 + 加 expiration_date/attributed_to/校验

### Task 2: search 增强

- 新增 `hybrid/reranker/recipe/explain/filters/forgetting/token_budget/inject_prompt/hyde/query_rewrite/show_expired` 参数
- 有高级参数时 → `to_thread(self._sync.search, ...)`
- 基础检索 (无高级参数) → 保留当前真 async 路径

### Task 3: update + delete 增强

- `update` 加 `user_id` 参数 → text_changed 时 `to_thread(self._sync.update, ...)` (含实体重链接)
- `delete` 加 `user_id` 参数 → 先 invalidate 再 delete (对齐 sync forget)

### Task 4: V2 编排方法

- `remember` → `to_thread(self._sync.remember, ...)`
- `recall` → `to_thread(self._sync.recall, ...)`
- `forget` → `to_thread(self._sync.forget, ...)`
- `improve` → `to_thread(self._sync.improve, ...)`

## 3. 不破坏的承诺

- 新增参数有默认值
- 基础 add/search 路径 (无高级参数) 保留真 async
- 现有方法签名向后兼容
