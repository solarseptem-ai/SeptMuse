# Embedding 差距补齐设计

> 日期：2026-08-06
> 前置：`docs/plans/optimization-roadmap-v2.md` Phase 0 基础设施优化（已完成 P0-Task 1/2/3/5/6 + Reranker 优化）
> 对标：`opensource/mem0/mem0/embeddings/`（15 个 embedding provider）
> 范围：全量对齐 mem0 的 embedding provider 列表 + memory_action 接口

---

## 1. 目标

SeptMuse 当前有 7 个 embedder 后端（hash/onnx/onnx-zh/bge-zh/auto/st/openai），mem0 有 11 个有效 provider。本次补齐缺失的 11 个 provider，并加 `memory_action` 接口参数，使 provider 列表 1:1 对齐 mem0（共 18 个后端，含 SeptMuse 独有的 5 个优势后端）。

**非目标**：不改 SeptMuse 独有优势后端（hash/onnx/bge-zh/auto/cached），不优化嵌入性能（已在 Phase 0 完成），不实现多模态嵌入。

## 2. 现状差距

### 2.1 Provider 数量差距

| Provider | mem0 | SeptMuse | 差距 |
|----------|------|----------|------|
| ollama | ✓ | ✗ | 缺：本地 Ollama 嵌入 |
| langchain | ✓ | ✗ | 缺：LangChain 桥接（通用适配器） |
| azure_openai | ✓ | ✗ | 缺：Azure OpenAI 嵌入 |
| huggingface | ✓ | ✗ | 缺：HuggingFace（本地 ST + TEI server 双模式） |
| gemini | ✓ | ✗ | 缺：Google Gemini 嵌入 |
| vertexai | ✓ | ✗ | 缺：Vertex AI 嵌入（task_type 区分 add/search） |
| together | ✓ | ✗ | 缺：Together AI 嵌入 |
| lmstudio | ✓ | ✗ | 缺：LM Studio 嵌入 |
| aws_bedrock | ✓ | ✗ | 缺：AWS Bedrock 嵌入 |
| fastembed | ✓ | ✗ | 缺：fastembed 库（轻量 ONNX） |
| mock | ✓ | ✗ | 缺：固定向量测试用 |

SeptMuse 独有优势（mem0 无）：hash（零配置测试）、onnx（无 torch ModelScope）、bge-zh（中文优化）、auto（语言检测）、cached（LRU 缓存）。

### 2.2 接口差距

mem0 的 `EmbeddingBase.embed(text, memory_action)` 支持 `"add"/"search"/"update"` 不同嵌入策略；SeptMuse 的 `Embedder.embed(text)` 不支持。实际只有 vertexai 真正用此参数区分 RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY，其他 provider 忽略。

## 3. 设计

### 3.1 memory_action 接口改造

Embedder ABC 签名改造（向后兼容，可选参数）：

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str, memory_action: str | None = None) -> list[float]: ...

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]
```

`memory_action` 取值：`"add"` / `"search"` / `"update"` / `None`。

现有 5 个 embedder（hash/onnx/auto/st/openai）签名加 `memory_action: str | None = None` 但内部忽略。仅 vertexai 真正使用此参数。

### 3.2 CachedEmbedder 透传

cache key 从 `text` 改为 `(text, memory_action)`，防止 vertexai 等 provider 的 add/search 向量混用。`embed`/`embed_batch` 透传 `memory_action` 给 inner embedder。

### 3.3 _OpenAICompatibleEmbedder 基类

提取现有 `OpenAIEmbedder` 的 embed/embed_batch 核心逻辑到基类：

- `client.embeddings.create()` 调用
- 100 批分块 + `.index` 排序
- 数量校验
- `\n` 替换为空格
- matryoshka `dimensions` 透传（仅当用户显式设 embedding_dims）

继承关系：

```
_OpenAICompatibleEmbedder (基类: embed/embed_batch 核心逻辑)
├── OpenAIEmbedder        (现有，重构为继承基类，__init__ 创建 OpenAI client)
├── TogetherEmbedder      (__init__ 创建 OpenAI client + base_url + together 默认 model/dims)
├── LMStudioEmbedder      (__init__ 创建 OpenAI client + base_url + lmstudio 默认 model/dims)
└── AzureOpenAIEmbedder   (__init__ 创建 AzureOpenAI client + AD token provider fallback)
```

together/lmstudio 不引独立 SDK（Together/LM Studio API 本就 OpenAI 兼容），减少依赖但 provider 列表对齐 mem0。

### 3.4 新 provider 实现细节

| # | Provider | 类名 | 文件 | 继承 | SDK | 默认 model | 默认 dims |
|---|----------|------|------|------|-----|-----------|----------|
| 1 | ollama | OllamaEmbedder | ollama.py | Embedder | `ollama` | nomic-embed-text | 512 |
| 2 | langchain | LangchainEmbedder | langchain.py | Embedder | `langchain` | (用户传入 Embeddings 实例) | (从 model 探测) |
| 3 | azure_openai | AzureOpenAIEmbedder | azure_openai.py | _OpenAICompatible | `openai`(AzureOpenAI) + `azure-identity` | (deployment) | 1536 |
| 4 | huggingface | HuggingFaceEmbedder | huggingface.py | Embedder | `sentence-transformers` + `openai`(TEI) | multi-qa-MiniLM-L6-cos-v1 | (探测) |
| 5 | gemini | GeminiEmbedder | gemini.py | Embedder | `google-genai` | gemini-embedding-001 | 768 |
| 6 | vertexai | VertexAIEmbedder | vertexai.py | Embedder | `google-cloud-aiplatform` | gemini-embedding-001 | 256 |
| 7 | together | TogetherEmbedder | together.py | _OpenAICompatible | `openai` + base_url | intfloat/multilingual-e5-large-instruct | 1024 |
| 8 | lmstudio | LMStudioEmbedder | lmstudio.py | _OpenAICompatible | `openai` + base_url | nomic-ai/nomic-embed-text-v1.5 | 1536 |
| 9 | aws_bedrock | AWSBedrockEmbedder | aws_bedrock.py | Embedder | `boto3` | amazon.titan-embed-text-v1 | (API 返回) |
| 10 | fastembed | FastEmbedEmbedder | fastembed.py | Embedder | `fastembed` | thenlper/gte-large | (探测) |
| 11 | mock | MockEmbedder | mock.py | Embedder | 无 | — | 10 |

关键设计点：

- **ollama**：`ollama.Client.embed()` 调用，自动 pull 模型（`_ensure_model_exists`），支持 `embed_batch` 真批量
- **langchain**：接收用户传入的 `langchain.embeddings.Embeddings` 实例，调 `embed_query()`/`embed_documents()`，无 SDK 依赖（用户自行安装 langchain）
- **azure_openai**：`AzureOpenAI` client，API key 或 `DefaultAzureCredential` AD token provider fallback
- **huggingface**：双模式 — 有 `huggingface_base_url` 时走 TEI server（OpenAI 兼容 API），无时走本地 `SentenceTransformer`
- **gemini**：`google.genai.Client.models.embed_content()`，支持 `output_dimensionality`
- **vertexai**：`TextEmbeddingModel.from_pretrained()`，唯一真正用 `memory_action` 的 provider（add→RETRIEVAL_DOCUMENT, search→RETRIEVAL_QUERY, 默认 SEMANTIC_SIMILARITY）
- **together/lmstudio**：继承 `_OpenAICompatibleEmbedder`，`__init__` 创建 OpenAI client + 厂商 base_url + 默认 model/dims
- **aws_bedrock**：`boto3.client("bedrock-runtime").invoke_model()`，按 provider（cohere/titan）构造不同 body，L2 归一化
- **fastembed**：`fastembed.TextEmbedding`，单条 embed（无原生 batch，`embed_batch` 逐条调）
- **mock**：固定 10 维向量 `[0.1, 0.2, ..., 1.0]`，确定性测试用

### 3.5 Config 设计

每个新 provider 有独立 config 子类（`src/septmuse/configs/embeddings/<provider>.py`），继承 `BaseEmbedderConfig`（pydantic），含 provider 特有字段：

| Provider | config_cls | 特有字段 |
|----------|-----------|---------|
| ollama | OllamaEmbedderConfig | ollama_base_url |
| langchain | LangchainEmbedderConfig | model（Embeddings 实例，非字符串） |
| azure_openai | AzureOpenAIEmbedderConfig | azure_deployment, azure_endpoint, api_version |
| huggingface | HuggingFaceEmbedderConfig | huggingface_base_url, model_kwargs |
| gemini | GeminiEmbedderConfig | output_dimensionality |
| vertexai | VertexAIEmbedderConfig | vertex_credentials_json, memory_add_embedding_type, memory_search_embedding_type, memory_update_embedding_type |
| together | TogetherEmbedderConfig | —（复用 api_key/base_url） |
| lmstudio | LMStudioEmbedderConfig | lmstudio_base_url |
| aws_bedrock | AWSBedrockEmbedderConfig | aws_access_key_id, aws_secret_access_key, aws_session_token, aws_region |
| fastembed | FastEmbedEmbedderConfig | — |
| mock | MockEmbedderConfig | — |

### 3.6 Registry 注册

`registry.py` `BACKEND_MANIFEST["embedder"]` 新增 11 条 BackendEntry：

```
hash / onnx / onnx-zh / bge-zh / auto / openai / st  (现有 7)
+ ollama / langchain / azure_openai / huggingface / gemini
+ vertexai / together / lmstudio / aws_bedrock / fastembed / mock  (新增 11)
= 18 个 embedder 后端
```

每条声明 module/cls/config_cls/deps。新 provider 全部走 `embedder_provider.resolve()` 默认路径（声明式实例化），不需要特殊 resolver 逻辑。仅 bge-zh/onnx-zh 保留现有特殊处理（model_name 覆盖）。

`EmbedderBackend` enum（`configs/enums.py`）新增 11 个值。

### 3.7 依赖分组（pyproject.toml extras）

| Provider | 库 | extra | 复用/新增 |
|----------|-----|-------|----------|
| ollama | `ollama` | `[ollama]` | 复用（LLM 已有） |
| gemini | `google-genai` | `[gemini]` | 复用（LLM 已有） |
| together/lmstudio | `openai` | `[openai]` | 复用（OpenAI 兼容） |
| huggingface | `sentence-transformers`+`openai` | `[st]` | 复用 |
| langchain | `langchain` | `[langchain]` | **新增** |
| azure_openai | `openai`+`azure-identity` | `[azure-openai]` | **新增** |
| vertexai | `google-cloud-aiplatform` | `[vertexai]` | **新增** |
| aws_bedrock | `boto3` | `[aws-bedrock]` | **新增** |
| fastembed | `fastembed` | `[fastembed]` | **新增** |
| mock | 无 | — | 无依赖 |

新增 5 个 extras。新增 `embedders` 聚合 extra（含 onnx/st/openai/ollama/gemini/langchain/azure-openai/vertexai/aws-bedrock/fastembed）。`all` 聚合 extra 同步更新。

## 4. 测试策略

### 4.1 双轨测试

每个新 provider 两类测试：

| 类型 | 做法 | 标记 |
|------|------|------|
| mock 单元测试 | `unittest.mock.patch` 注入假 SDK client，验证 embed/embed_batch 签名、memory_action 透传、batch 分块、数量校验、默认 model/dims、ImportError 降级 | 无标记（默认跑） |
| integration 测试 | 真实 API key / 服务 | `@pytest.mark.integration`（skipped 默认） |

### 4.2 新增测试文件

- `tests/unit/test_embedders/test_ollama.py` — mock `ollama.Client`，验证 embed/embed_batch/pull
- `tests/unit/test_embedders/test_langchain.py` — 用 langchain `FakeEmbeddings`，验证 embed_query 透传
- `tests/unit/test_embedders/test_azure_openai.py` — mock `AzureOpenAI`，验证 AD token fallback
- `tests/unit/test_embedders/test_huggingface.py` — mock `SentenceTransformer` + TEI 双模式
- `tests/unit/test_embedders/test_gemini.py` — mock `genai.Client`，验证 embed_content
- `tests/unit/test_embedders/test_vertexai.py` — mock `TextEmbeddingModel`，验证 memory_action task_type 切换
- `tests/unit/test_embedders/test_together.py` — mock OpenAI client，验证 base_url + 默认 model/dims
- `tests/unit/test_embedders/test_lmstudio.py` — mock OpenAI client，验证 base_url + 默认 model/dims
- `tests/unit/test_embedders/test_aws_bedrock.py` — mock `boto3.client`，验证 cohere/titan body 差异 + L2 归一化
- `tests/unit/test_embedders/test_fastembed.py` — mock `TextEmbedding`，验证 embed
- `tests/unit/test_embedders/test_mock.py` — 固定向量验证
- `tests/unit/test_embedders/test_memory_action.py` — ABC 接口 + CachedEmbedder cache key 隔离 + 现有 embedder 向后兼容

### 4.3 现有测试兼容

Embedder ABC 签名加 `memory_action=None`（可选参数），现有 `embed(text)` 调用不传参仍兼容。CachedEmbedder 透传 `memory_action`，cache key 改为 `(text, memory_action)`。**零退化**：现有 1319 passed + 23 skipped 基线不变。

### 4.4 _OpenAICompatibleEmbedder 基类测试

重构现有 `OpenAIEmbedder` 为继承基类后，现有 `test_embedder.py` 中 OpenAI 测试全部通过（验证重构无行为变化）。新增基类专项测试验证 batch 分块 + index 排序 + matryoshka dimensions 透传逻辑。

## 5. 实现清单

### 5.1 接口改造（7 文件改）

1. `src/septmuse/embedders/base.py` — Embedder ABC 加 `memory_action` 参数
2. `src/septmuse/embedders/hash.py` — HashEmbedder 签名加 `memory_action=None`
3. `src/septmuse/embedders/onnx.py` — OnnxEmbedder 签名加 `memory_action=None`
4. `src/septmuse/embedders/auto.py` — AutoOnnxEmbedder 透传 `memory_action`
5. `src/septmuse/embedders/sentence_transformers.py` — SentenceTransformerEmbedder 签名加 `memory_action=None`
6. `src/septmuse/embedders/cached.py` — CachedEmbedder 透传 + cache key 改为 `(text, memory_action)`
7. `src/septmuse/embedders/openai.py` — 重构为继承 `_OpenAICompatibleEmbedder`

### 5.2 新增基类（1 文件）

8. `src/septmuse/embedders/_openai_compatible.py` — `_OpenAICompatibleEmbedder` 基类

### 5.3 新增 11 个 provider（11 embedder + 11 config = 22 文件）

9-19. `src/septmuse/embedders/{ollama,langchain,azure_openai,huggingface,gemini,vertexai,together,lmstudio,aws_bedrock,fastembed,mock}.py`
20-30. `src/septmuse/configs/embeddings/{ollama,langchain,azure_openai,huggingface,gemini,vertexai,together,lmstudio,aws_bedrock,fastembed,mock}.py`

### 5.4 注册 + 枚举（2 文件改）

31. `src/septmuse/services/registry.py` — BACKEND_MANIFEST["embedder"] 新增 11 条
32. `src/septmuse/configs/enums.py` — EmbedderBackend enum 新增 11 个值

### 5.5 测试（12 文件）

33-44. `tests/unit/test_embedders/test_*.py`（11 个 provider + 1 个 memory_action 接口）

### 5.6 依赖 + 文档（3 文件改）

45. `pyproject.toml` — 新增 5 extras + embedders 聚合 + all 更新
46. `AGENTS.md` — Embedder section 更新
47. `CHANGELOG.md` — 记录变更

## 6. 风险与权衡

| 风险 | 缓解 |
|------|------|
| ABC 签名变更破坏现有 embedder | `memory_action=None` 可选参数，现有调用不传参仍兼容 |
| CachedEmbedder cache key 变更影响缓存命中 | cache key 改为 `(text, memory_action)`，大多数 provider memory_action=None，命中行为不变 |
| OpenAIEmbedder 重构为继承基类引入 bug | 基类逻辑从现有 OpenAIEmbedder 提取，行为不变，现有测试覆盖验证 |
| langchain config 的 model 字段是 Embeddings 实例（非字符串） | config 用 `Any` 类型，registry 实例化时 ServiceProvider._config_to_kwargs 处理 |
| 大量新依赖膨胀 | 全部走 extras，核心零依赖不变（hash/bge-zh 默认） |
