# LLM Provider 设计文档（P3-Task 1）

> 日期：2026-07-24
> 前置文档：`docs/plans/development-roadmap.md`（P3 Phase）
> 范围：P3-Task 1（LLM Provider 实现 — openai/ollama/anthropic/dashscope + _resolve_llm 工厂）
> 不包含：P3-Task 2（单次 LLM 事实抽取）、P3-Task 3（冲突解决）、P3-Task 4（session 蒸馏）、P3-Task 5（自编辑）
> 借鉴来源：mem0 `OpenAILLM`/`OllamaLLM` + MemOS `AnthropicLlmProvider` + SeptMuse `_resolve_embedder`/`_resolve_reranker` 工厂模式

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM Provider 架构                          │
│                                                              │
│  SEPTMUSE_LLM=openai  →  _resolve_llm(config)               │
│                           │                                  │
│                           ├── openai.py   OpenAILLM          │
│                           ├── ollama.py   OllamaLLM          │
│                           ├── anthropic.py AnthropicLLM      │
│                           └── dashscope.py DashScopeLLM     │
│                                                              │
│  每个 provider:                                              │
│    - 延迟 import SDK（仅使用时加载）                         │
│    - API key 从环境变量读                                    │
│    - 实现 LLM ABC: complete(system_prompt, user_prompt) -> str│
│    - JSON 输出靠 prompt 工程（调用方追加 JSON 指令）        │
│                                                              │
│  Memory.__init__:                                            │
│    llm 参数注入 → 用注入的                                    │
│    llm=None + config.llm_provider → _resolve_llm(config)     │
│    llm=None + llm_provider=None → None (verbatim 模式)       │
└─────────────────────────────────────────────────────────────┘
```

### 关键决策

- **ABC 不变**：`complete(system_prompt, user_prompt) -> str`，JSON 靠 prompt 工程（调用方追加 "respond in JSON only" 指令，自己 `json.loads`）。所有 provider 通用，向后兼容。
- **4 个 provider**：`openai`/`ollama`/`anthropic`/`dashscope`，各一个文件。
- **延迟 import**：每个 provider 在 `__init__` 中 import SDK，不用就不加载（对齐 embedder/reranker 策略）。
- **`_resolve_llm` 工厂**：`SEPTMUSE_LLM` 环境变量 → provider 实例（对齐 `_resolve_embedder`/`_resolve_reranker` 模式）。
- **零配置优先**：`ollama` 可本地零配置运行（`OLLAMA_HOST` 默认 localhost:11434，无需 API key）。
- **Mock 测试**：不调真实 API，monkeypatch SDK 返回固定文本。

---

## 2. Provider 实现

### 2.1 通用模式

每个 provider 文件结构一致：

```python
from septmuse.providers.llms.base import LLM

class XxxLLM(LLM):
    def __init__(self, *, model: str = "...", api_key: str | None = None, base_url: str | None = None):
        # 延迟 import
        import xxx_sdk
        self._client = xxx_sdk.Client(api_key=..., base_url=...)
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat(...)
        return response.text
```

### 2.2 OpenAILLM

借鉴 mem0 `OpenAILLM`。

- SDK: `openai>=1.0`（`from openai import OpenAI`）
- 环境变量: `OPENAI_API_KEY`（必须）、`OPENAI_BASE_URL`（可选，支持 OpenAI 兼容端点如 vLLM/LiteLLM）
- 默认模型: `gpt-4o-mini`
- 调用: `client.chat.completions.create(model, messages=[{system}, {user}])`
- 返回: `response.choices[0].message.content`
- 可选 extra: `pip install septmuse[openai]`

### 2.3 OllamaLLM

借鉴 mem0 `OllamaLLM`。

- SDK: `ollama>=0.1`（`from ollama import Client`）
- 环境变量: `OLLAMA_HOST`（默认 `http://localhost:11434`，无需 API key）
- 默认模型: `qwen2.5:7b`
- 调用: `client.chat(model, messages=[{system}, {user}], options={temperature, num_predict})`
- 返回: `response["message"]["content"]`
- 零配置本地运行
- 可选 extra: `pip install septmuse[ollama]`

### 2.4 AnthropicLLM

借鉴 MemOS `AnthropicLlmProvider`。

- SDK: `anthropic>=0.20`（`from anthropic import Anthropic`）
- 环境变量: `ANTHROPIC_API_KEY`（必须）
- 默认模型: `claude-3-5-haiku-latest`
- 调用: `client.messages.create(model, system=system_prompt, messages=[{user}], max_tokens=4096)`
- **system 消息单独传**（Anthropic API 要求）
- 返回: `response.content[0].text`
- 可选 extra: `pip install septmuse[anthropic]`

### 2.5 DashScopeLLM

SeptMuse 创新（对齐中国用户）。

- SDK: `dashscope>=1.17`（`import dashscope`）
- 环境变量: `DASHSCOPE_API_KEY`（必须）
- 默认模型: `qwen-plus`
- 调用: `dashscope.Generation.call(model, messages=[{system}, {user}], api_key=key, result_format="message")`
- 返回: `response.output.choices[0].message.content`
- 可选 extra: `pip install septmuse[dashscope]`

---

## 3. _resolve_llm 工厂 + Memory 集成

### 3.1 `_resolve_llm` 工厂函数

```python
# providers/llms/__init__.py

def _resolve_llm(config: MemoryConfig) -> LLM | None:
    """工厂函数: 根据 config.llm_provider 创建 LLM 实例。

    llm_provider=None → 返回 None (verbatim 模式)
    """
    provider = config.llm_provider
    if provider is None:
        return None

    match provider:
        case "openai":
            from septmuse.providers.llms.openai import OpenAILLM
            return OpenAILLM(model=config.llm_model or "gpt-4o-mini")
        case "ollama":
            from septmuse.providers.llms.ollama import OllamaLLM
            return OllamaLLM(model=config.llm_model or "qwen2.5:7b")
        case "anthropic":
            from septmuse.providers.llms.anthropic import AnthropicLLM
            return AnthropicLLM(model=config.llm_model or "claude-3-5-haiku-latest")
        case "dashscope":
            from septmuse.providers.llms.dashscope import DashScopeLLM
            return DashScopeLLM(model=config.llm_model or "qwen-plus")
        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
```

### 3.2 `MemoryConfig` 扩展

```python
class MemoryConfig(BaseModel):
    # ... 已有字段 ...
    llm_model: str | None = Field(
        default=None,
        description="LLM 模型名 (None → provider 默认模型)",
    )
    # llm_provider 已存在 (default=None)
```

### 3.3 `Memory.__init__` 修改

当前逻辑（已有）：
```python
self.llm: LLM | None = llm
self.extractor: FactExtractor | None = None
if self.llm is not None:
    self.extractor = FactExtractor(self.llm, ...)
```

新增：当 `llm` 参数为 None 但 `config.llm_provider` 已设时，用 `_resolve_llm` 自动创建：

```python
self.llm: LLM | None = llm
if self.llm is None and self.config.llm_provider is not None:
    self.llm = _resolve_llm(self.config)
self.extractor: FactExtractor | None = None
if self.llm is not None:
    self.extractor = FactExtractor(self.llm, ...)
```

### 3.4 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_LLM` | 未设 | `openai`/`ollama`/`anthropic`/`dashscope`；未设=verbatim 模式 |
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |

`default_config()` 新增：
```python
llm_model=os.getenv("SEPTMUSE_LLM_MODEL"),
```

### 3.5 pyproject.toml extras

```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
ollama = ["ollama>=0.1"]
anthropic = ["anthropic>=0.20"]
dashscope = ["dashscope>=1.17"]
llm = ["openai>=1.0"]
```

---

## 4. 测试策略

### 4.1 测试文件布局

| 文件 | 内容 | 预计测试数 |
|------|------|-----------|
| `tests/unit/test_llm_providers.py` | 4 provider mock 测试 + `_resolve_llm` 工厂 | ~22 |
| `tests/e2e/test_llm_e2e.py` | e2e：ollama 集成（skip if 不可用） | 2 |

### 4.2 测试要点

**Provider 单元测试**（`test_llm_providers.py`）：
- **OpenAILLM**：mock `openai.OpenAI` → `complete()` 返回 mock 文本；API key 从环境变量；`base_url` 传参
- **OllamaLLM**：mock `ollama.Client` → `complete()` 返回 mock 文本；默认 host `localhost:11434`
- **AnthropicLLM**：mock `anthropic.Anthropic` → `complete()` 返回 mock 文本；system prompt 单独传
- **DashScopeLLM**：mock `dashscope.Generation.call` → `complete()` 返回 mock 文本
- **`_resolve_llm`**：4 种 provider 正确分派；`None` 返回 None；未知 provider 抛 ValueError

**Mock 策略**（借鉴现有 `test_reranker.py` 的 `_MockLLM` 模式）：
- 用 `monkeypatch` 替换 SDK 客户端类，返回固定文本
- 不调真实 API（零成本 + 零网络依赖）
- 验证 SDK 调用参数（model、messages 格式、system 分离等）

**e2e 测试**（`test_llm_e2e.py`）：
- `@pytest.mark.integration` 标记，`ollama` 不可用时 skip
- 测试 `SEPTMUSE_LLM=ollama` 端到端：`Memory.add(infer=True)` → FactExtractor 抽取
- 测试 LLMReranker 端到端：`Memory.search(reranker="llm")` 用 LLM 重排

### 4.3 验收标准

- 现有 828 passed 测试零回归
- 新增 ~24 测试 → 总计 ~852 passed
- `ruff check` + `ruff format --check` clean
- SDK 不可用时 skip（非 fail），对齐 embedder/reranker skip 模式

---

## 5. 文件变更清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `src/septmuse/providers/llms/openai.py` | OpenAILLM |
| `src/septmuse/providers/llms/ollama.py` | OllamaLLM |
| `src/septmuse/providers/llms/anthropic.py` | AnthropicLLM |
| `src/septmuse/providers/llms/dashscope.py` | DashScopeLLM |
| `tests/unit/test_llm_providers.py` | ~22 单元测试 |
| `tests/e2e/test_llm_e2e.py` | 2 e2e 测试 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/septmuse/providers/llms/__init__.py` | `_resolve_llm` 工厂函数 |
| `src/septmuse/orchestration/memory.py` | `__init__` 用 `_resolve_llm` 自动创建 |
| `src/septmuse/configs/defaults.py` | `MemoryConfig` +`llm_model` + `default_config()` 读 `SEPTMUSE_LLM_MODEL` |
| `pyproject.toml` | +`openai`/`ollama`/`anthropic`/`dashscope`/`llm` extras |
| `CHANGELOG.md` | Added — LLM Provider |
| `AGENTS.md` | +SEPTMUSE_LLM_MODEL 环境变量 + LLM Provider 章节 |

---

## 6. 借鉴来源映射

| 设计要素 | 借鉴来源 | 具体文件/类 |
|----------|----------|------------|
| OpenAILLM | mem0 `OpenAILLM` | `opensource/mem0/mem0/llms/openai.py:14` |
| OllamaLLM | mem0 `OllamaLLM` | `opensource/mem0/mem0/llms/ollama.py:15` |
| AnthropicLLM system 分离 | MemOS `AnthropicLlmProvider` | `opensource/MemOS/.../anthropic.ts:26` |
| DashScopeLLM | SeptMuse 创新（对齐中国用户） | — |
| `_resolve_llm` 工厂 | SeptMuse `_resolve_embedder`/`_resolve_reranker` | `src/septmuse/providers/embedders/__init__.py` |
| 延迟 import + 环境变量 | SeptMuse embedder/reranker 模式 | `src/septmuse/concerns/retrieval/reranker.py` |
| Mock 测试策略 | SeptMuse `test_reranker.py` `_MockLLM` | `tests/unit/test_reranker.py` |

---

## 7. 不包含（Out of Scope）

- **P3-Task 2 单次 LLM 事实抽取**：依赖本 Task 的 LLM Provider。重构 `FactExtractor` 为单次 LLM 调用 + ADDITIVE_EXTRACTION_PROMPT。
- **P3-Task 3 冲突解决**：依赖 P0-Task 2（三元组抽取）+ P2-Task 1（双时态）。LLM 矛盾检测 + 自动失效。
- **P3-Task 4 session 蒸馏**：依赖本 Task。两阶段 LLM（curator → writer/rejecter）。
- **P3-Task 5 自编辑**：依赖本 Task。`memory_apply_patch` + `memory_rethink`。
- **P0-Task 2/3 三元组抽取 + cognify**：依赖本 Task。LLM 联合抽取实体+边 + 流水线。
- **P1-Task 3/4 BFS 图遍历 + Recipes**：依赖 P0-Task 3。
- **structured_output / JSON schema 约束**：当前用 prompt 工程。未来可扩展 ABC 加 `response_format` 参数。
- **流式输出 (streaming)**：当前 `complete()` 是同步阻塞。流式留给后续。
