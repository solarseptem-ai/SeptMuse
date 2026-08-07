# Task 7 Report: recall 画像注入 + REST 端点

## Status: DONE

## 一行总结
实现 `_profile_to_prompt` 画像摘要生成 + `GET /agents/{user_id}/profile` REST 端点，7 新测试全通过，19 回归测试全通过，ruff 清洁。

## 改动文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/septmuse/memory/main.py` | 修改 | 实现 `_profile_to_prompt` (line 1037): 从占位返回 "" 改为生成 attributes/preferences/relationships/plans 摘要串 |
| `src/septmuse/api/rest/__init__.py` | 修改 | 新增 `GET /agents/{user_id}/profile` 端点 (line 414): 支持 `include_temporal` 查询参数 |
| `tests/unit/test_recall_profile_inject.py` | 新建 | 4 测试: 画像注入/默认不注入/空用户/格式校验 |
| `tests/unit/test_rest_profile.py` | 新建 | 3 测试: 基础查询/时态查询/空用户 |

## 测试结果

### 新测试 (TDD)
```
tests/unit/test_recall_profile_inject.py: 4 passed
tests/unit/test_rest_profile.py: 3 passed
合计: 7 passed in 3.09s
```

### 回归测试
```
tests/unit/test_memory_orchestration.py + tests/unit/test_user_profile.py: 19 passed
```

### Lint
```
ruff check src/septmuse/memory/main.py src/septmuse/api/rest/__init__.py tests/unit/test_recall_profile_inject.py tests/unit/test_rest_profile.py
→ All checks passed!
```

## 实现细节

### `_profile_to_prompt` (main.py:1037)
- 遍历 `profile.attributes`，过滤 `is_current=True` 的值，生成 `- Name: value` 格式
- preferences 聚合为一行 `- Current preferences: val1, val2`
- relationships 聚合为 `- Relationships: key: val; ...`
- plans 聚合为 `- Plans: val; ...`
- 仅当有画像内容 (lines > 1) 时返回非空串，避免空画像注入噪声
- `recall(inject_profile=True)` 已在 Task 4 调用此方法 (try/except 容错)，实现后自动生效

### REST 端点 (rest/__init__.py:414)
- `GET /agents/{user_id}/profile?include_temporal=true|false`
- 通过 `app.state.memory.get_user_profile()` 聚合 (ExperimentalMemory 继承 Memory)
- 返回结构化 JSON: attributes/preferences/relationships/plans/raw_facts/temporal_summary
- 放在 `/agents/{user_id}/memories` 端点之后，复用 `sharing` tag

## 顾虑
- 无。REST 测试 fixture 通过 `SEPTMUSE_DB_PATH` 环境变量 + 独立 engine 共享同一 DB 文件，`MemoryConfig._flat_env_aliases` 确保环境变量正确映射到 `database.db_path`。
