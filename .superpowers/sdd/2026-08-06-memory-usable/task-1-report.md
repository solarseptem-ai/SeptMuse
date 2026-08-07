# Task 1 报告 — ADDITIVE_DECISION_PROMPT + Decision + extract_with_decisions

## Status: DONE

## 实现内容

### 修改文件

**`src/septmuse/prompts/extract.py`**
- 新增 `ADDITIVE_DECISION_PROMPT` 常量（文件末尾追加）— 对齐 mem0 additive decision prompt，四决策 ADD/UPDATE/DELETE/NOOP + confidence，含 4 个 few-shot 示例。

**`src/septmuse/models/extract.py`**
- 新增 `from dataclasses import dataclass` 导入。
- 新增 `Decision` dataclass（`text`/`event`/`id`/`confidence` 四字段，位于 `fact_to_triple` 与 `FactExtractor` 之间）。
- 新增 `FactExtractor.extract_with_decisions(messages, existing_memories)` 方法 — LLM 决策抽取，空文本短路返回空列表，复用 `parse_messages` + `build_extraction_user_prompt`。
- 新增 `FactExtractor._parse_decisions_response(raw)` staticmethod — 容错解析（去 markdown 代码块 → JSON 解析失败降级空列表 → 非法 event 过滤 → 非 dict 跳过）。

### 新建文件

**`tests/unit/test_fact_decision.py`** — 6 个测试：
1. `test_decision_dataclass` — Decision 默认值
2. `test_extract_with_decisions_add` — ADD 决策 + confidence
3. `test_extract_with_decisions_four_events` — 四决策齐全 + id/confidence 字段
4. `test_extract_with_decisions_parse_fallback` — 非 JSON 降级空列表
5. `test_extract_with_decisions_empty_text` — 空文本短路
6. `test_extract_with_decisions_invalid_event_filtered` — 非法 event 过滤

## 测试结果

```
tests/unit/test_fact_decision.py: 6 passed in 2.52s
```

TDD 流程已验证：
- RED 阶段：`ImportError: cannot import name 'Decision' from 'septmuse.models.extract'`（实现前）
- GREEN 阶段：实现后 6 passed

### 回归验证（现有 extract 相关测试）

```
tests/unit/test_fact_extraction.py tests/unit/test_extract.py tests/unit/test_incremental_extraction.py: 48 passed in 10.40s
```

无回归（输出中的 `QdrantClient.__del__` / `ImportError: sys.meta_path is None` 为解释器关闭期噪声，与本改动无关）。

## Lint 结果

```
$ ruff check src/septmuse/prompts/extract.py src/septmuse/models/extract.py tests/unit/test_fact_decision.py
All checks passed!
```

未运行 `ruff format`（AGENTS.md：Windows 上有清空文件 bug，仅 `ruff check`）。

## 与规格的偏差（说明）

1. **测试文件移除未使用的 `MagicMock` 导入**：任务提供的测试代码含 `from unittest.mock import MagicMock` 但未使用，会触发 ruff F401。已移除（测试语义不变，仅清洁化新文件）。
2. **`_parse_decisions_response` 使用模块顶层 `json`/`re` 导入**：`models/extract.py` 顶部已 `import json` / `import re`，故未在方法内重复 local import（规格中给出的代码片段含局部 `import re`/`import json`），保持代码风格一致并避免冗余绑定。

## 顾虑

- 无功能顾虑。`extract_with_decisions` 与现有 `extract_facts`/`extract_and_store` 完全解耦，未改动后两者（属 Task 2 范围）。`_MockLLM` 以 duck typing 满足 `FactExtractor(llm=...)`，未继承 `LLM` ABC，但 `complete(system_prompt, user_prompt)` 签名一致。
- 构造函数、`extract_facts`、`extract_and_store`、`_retrieve_existing`、`_parse_facts_response` 均未改动。
