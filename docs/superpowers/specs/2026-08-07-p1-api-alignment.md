# P1: Memory API 对齐设计

> 日期: 2026-08-07
> 状态: 待审阅
> 范围: `src/septmuse/memory/main.py` + `src/septmuse/prompts/extract.py` + `src/septmuse/models/extract.py` + `src/septmuse/storage/base.py`

## 1. 背景

P0 完成后,Memory.add 决策透传、上下文窗口、实体批量链接、update 重链接已对齐 mem0 V3。P1 补 4 项 API 层差距:

| # | 差距 | mem0 对应 |
|---|------|-----------|
| 1 | `linked_memory_ids` 在 prompt 输出 | ADDITIVE_DECISION_PROMPT 缺 `linked_memory_ids` 字段 |
| 2 | `expiration_date` 过期 | add 存入 metadata, search/get_all 过滤 |
| 3 | 输入校验 | `_validate_and_trim_entity_id` + `_validate_search_params` |
| 4 | `reset()` 方法 | 删除 collection + 重建 |

## 2. 改造

### Task 1: linked_memory_ids prompt 输出

**现状**: `ADDITIVE_DECISION_PROMPT` 输出 `{"text","event","id","confidence"}`,无 `linked_memory_ids`。LLM 不输出跨记忆链接。

**改造**:
- `ADDITIVE_DECISION_PROMPT` 加 `linked_memory_ids` 字段说明 + 示例
- `Decision` dataclass 加 `linked_memory_ids: list[str]` 字段
- `_parse_decisions_response` 解析 `linked_memory_ids`
- `extract_and_store` ADD 路径把 `linked_memory_ids` 存入 verbatim metadata

**prompt 改动**:
```
# Output Format
Return ONLY valid JSON: {"facts": [{"text": "...", "event": "ADD|UPDATE|DELETE|NOOP", "id": null_or_memid, "confidence": 0.0-1.0, "linked_memory_ids": ["memid1", ...]}]}
- linked_memory_ids: IDs of existing memories related to this new fact (same topic, updated preference, follow-up). Empty array if none.
```

### Task 2: expiration_date 过期

**现状**: 无过期机制。mem0 支持 `expiration_date` (YYYY-MM-DD),过期记忆自动隐藏。

**改造**:
- `Memory.add` 加 `expiration_date: str | None = None` 参数
- 存入 metadata["expiration_date"]
- `Memory.search` 加 `show_expired: bool = False` 参数,过滤过期
- `Memory.get_all` 加 `show_expired: bool = False` 参数
- 过滤逻辑: `payload["expiration_date"] < today` → skip

**辅助**: `_is_expired(metadata) -> bool` + `_normalize_expiration_date(value)`

### Task 3: 输入校验

**现状**: 无校验,空 user_id/空 query/负 top_k 直接传到底层。

**改造**: 新增 `src/septmuse/core/validation.py`:
- `validate_entity_id(value, name) -> str` — trim + 拒空 + 拒内部空格
- `validate_search_params(threshold, top_k)` — threshold [0,1], top_k 非负整数
- `validate_search_query(query) -> str` — trim + 拒空

`Memory.add/search/get_all/delete_all` 入口调校验,不通过抛 `ValueError`。

### Task 4: reset() 方法

**现状**: 无重置方法。mem0 `reset()` 删除 collection + 重建。

**改造**: `Memory.reset()`:
- `self.store.reset()` (ORMMemoryStore 新增,清表 + 重建)
- `self.entity_store` 清表 (如果存在)
- `self.typed_store` 清表
- `self._invalidate_search_cache()`

## 3. 不破坏的承诺

- 新增参数有默认值,向后兼容
- `Decision.linked_memory_ids` 默认 `[]`
- `show_expired=False` 默认 (不破坏现有行为)
- 输入校验只在入口层,不影响内部调用
- `reset()` 是新方法,不改现有

## 4. 测试

| 文件 | 覆盖 |
|------|------|
| `test_linked_memory_ids.py` | prompt 含 linked_memory_ids + Decision 解析 + metadata 存储 |
| `test_expiration.py` | add 过期日期 + search 过滤 + get_all 过滤 + show_expired |
| `test_validation.py` | entity_id 校验 + search params 校验 + query 校验 |
| `test_reset.py` | reset 清空 + 重建 + 不崩 |
