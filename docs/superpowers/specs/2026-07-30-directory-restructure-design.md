# SeptMuse 目录结构优化设计

> 日期：2026-07-30
> 方案：B（中度重组，扁平直白型）
> 状态：已确认，待写实施计划

## 1. 目标

将 SeptMuse 的 `src/septmuse/` 目录从"学术化分层 + 顶层碎片化"重构为"扁平直白"风格，消除抽象目录名（concerns/orchestration/providers/content_types/metacognition），合并职责重叠目录（schemas vs content_types），治理顶层碎片（exceptions/types/utils/observability 各 1 文件）。

### 非目标

- 不重构 `configs/`（刚完成组合式重构，41 文件结构合理）
- 不重构 `storage/`（27 文件，目录名已直白）
- 不删减功能或合并实现文件，仅移动文件位置 + 重命名目录
- 不改业务逻辑，纯目录搬迁 + import 路径更新

## 2. 当前结构问题诊断

当前 172 个 .py 文件，16 个顶层目录：

| 问题 | 症状 |
|------|------|
| 配置/实现双轨散落 | `configs/vector_stores/qdrant.py` ↔ `storage/vector/qdrant.py` 同名散落两处（本次不动，仅记录） |
| concerns 命名抽象 | `concerns/`（"关注点"）无业务语义，新人难猜 `septmuse.concerns.retrieval` |
| orchestration 名不副实 | "编排"实际是 Memory facade + 借鉴 MemOS 容器 |
| providers vs services 重叠 | `providers/`(实现) 和新建的 `services/`(框架) 概念重叠 |
| schemas vs content_types 边界模糊 | schemas/ 是纯数据模型，content_types/ 是操作行为，同名 block 散落两处 |
| 顶层碎片化 | exceptions(1)/types(1)/utils(1)/observability(2) 各占独立目录 |
| compression 过度细分 | 1 个 summarizer.py 独占目录 |
| 命名学术化 | metacognition/evolution/compression 新人难猜 |

## 3. 目标结构

```
septmuse/
├── __init__.py              ← 顶层导出 Memory + default_config
├── experimental.py          ← 保留（ExperimentalMemory 继承 Memory）
│
├── memory/                  ← 原 orchestration/ 重命名
│   ├── __init__.py          ←   re-export Memory / MemCube / MemoryRegistry
│   ├── main.py              ←   原 memory.py（Memory facade，15 核心方法）
│   ├── cube.py              ←   原 mem_cube.py（MemCube 统一容器）
│   ├── os.py                ←   原 mem_os.py（MemOS 借鉴）
│   └── registry.py          ←   原 registry.py（MemoryRegistry）
│
├── retrieval/               ← 原 concerns/retrieval（9 files）
│                              hybrid/graph_search/progressive/recipes/
│                              reranker/temporal/forgetting/causal
├── governance/              ← 原 concerns/governance + concerns/sharing 合并
│                              permissions/privacy/access_log/approval/
│                              degradation/token_budget + rbac/user_id
├── extraction/              ← 原 concerns/extraction（triplet/entity/cognify）
├── evolution/               ← 原 concerns/evolution + concerns/compression 合并
│                              conflict/reflect/dream/zettel + summarizer
├── capture/                 ← 原 concerns/capture（hooks/pipeline）
├── meta/                    ← 原 concerns/metacognition 改名（coverage/router/strategy）
│
├── embedders/               ← 原 providers/embedders（hash/onnx/openai/auto/langdetect/sentence_transformers）
├── llms/                    ← 原 providers/llms（openai/ollama/anthropic/dashscope/mock）
├── rerankers/               ← 原 providers/rerankers
│
├── services/                ← 保留（Service/Factory/Manager/deps/schema）
├── storage/                 ← 保留（vector/keyword/graph/sqlite/file/parametric + activation + entity_store + typed_store）
├── configs/                 ← 保留（刚重构，7 子目录组合式配置）
│
├── models/                  ← content_types + schemas 合并
│   ├── block.py             ←   schemas/block（Block 数据模型）+ content_types/working/block（WorkingMemory 操作）
│   ├── episodic.py          ←   schemas/episodic + content_types/episodic
│   ├── semantic.py          ←   schemas/semantic
│   ├── procedural.py        ←   schemas/procedural
│   ├── causal.py            ←   schemas/causal
│   ├── strength.py          ←   schemas/strength
│   ├── extract.py           ←   content_types/semantic/extract（LLM 事实抽取）
│   └── fact.py              ←   content_types/semantic/fact
│
├── core/                    ← exceptions + types + utils + observability 合并
│   ├── exceptions.py        ←   原 exceptions/__init__.py
│   ├── types.py             ←   原 types/__init__.py
│   ├── utils.py             ←   原 utils/__init__.py
│   └── logging.py           ←   原 observability/logging_utils.py
│
├── api/                     ← 保留（rest/mcp/auth）
├── cli/                     ← 保留（main.py）
├── prompts/                 ← 保留（extract 等 prompt 模板）
└── sync/                    ← 保留（synchronizer/drift）
```

**顶层目录数：16 → 19**（concerns 拍平 +7、providers 拆解 +3、合并 -6、orchestration 替换 +0）。每个目录名直白，无抽象名。

## 4. 已确认决策点

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 1 | sharing 归属 | 合入 governance/ | RBAC 与 permissions 概念重叠，合并后 governance 涵盖所有"权限/隔离/审计" |
| 2 | metacognition 归属 | 独立改名为 meta/ | router/coverage/strategy 是系统自省，与 retrieval 的"执行检索"职责不同；meta 短名直白 |
| 3 | 向后兼容 | re-export 兼容层 | SeptMuse 要 pip install，旧路径保留 re-export，下个 major 版本删除 |

## 5. 向后兼容策略

旧 import 路径保留 re-export，确保 `pip install septmuse` 的外部代码不断：

### 兼容 re-export 清单

| 旧路径 | 新路径 | 兼容方式 |
|--------|--------|----------|
| `septmuse.concerns.retrieval.X` | `septmuse.retrieval.X` | `concerns/retrieval/__init__.py` → `from septmuse.retrieval import *` |
| `septmuse.concerns.governance.X` | `septmuse.governance.X` | 同上 |
| `septmuse.concerns.extraction.X` | `septmuse.extraction.X` | 同上 |
| `septmuse.concerns.evolution.X` | `septmuse.evolution.X` | 同上 |
| `septmuse.concerns.capture.X` | `septmuse.capture.X` | 同上 |
| `septmuse.concerns.metacognition.X` | `septmuse.meta.X` | `concerns/metacognition/__init__.py` → `from septmuse.meta import *` |
| `septmuse.concerns.compression.X` | `septmuse.evolution.X` | `concerns/compression/__init__.py` → `from septmuse.evolution.summarizer import *` |
| `septmuse.concerns.sharing.X` | `septmuse.governance.X` | `concerns/sharing/__init__.py` → `from septmuse.governance import *` |
| `septmuse.orchestration.memory.Memory` | `septmuse.memory.main.Memory` | `orchestration/__init__.py` → `from septmuse.memory import *` |
| `septmuse.orchestration.mem_cube.X` | `septmuse.memory.cube.X` | 同上 |
| `septmuse.orchestration.mem_os.X` | `septmuse.memory.os.X` | 同上 |
| `septmuse.orchestration.registry.X` | `septmuse.memory.registry.X` | 同上 |
| `septmuse.providers.embedders.X` | `septmuse.embedders.X` | `providers/embedders/__init__.py` → `from septmuse.embedders import *` |
| `septmuse.providers.llms.X` | `septmuse.llms.X` | 同上 |
| `septmuse.providers.rerankers.X` | `septmuse.rerankers.X` | 同上 |
| `septmuse.schemas.X` | `septmuse.models.X` | `schemas/__init__.py` → `from septmuse.models import *` |
| `septmuse.content_types.X` | `septmuse.models.X` | `content_types/__init__.py` → `from septmuse.models import *` |
| `septmuse.exceptions.X` | `septmuse.core.exceptions.X` | `exceptions/__init__.py` → `from septmuse.core.exceptions import *` |
| `septmuse.types.X` | `septmuse.core.types.X` | 同上 |
| `septmuse.utils.X` | `septmuse.core.utils.X` | 同上 |
| `septmuse.observability.X` | `septmuse.core.logging.X` | 同上 |

### 兼容层标记

- 所有兼容 re-export 模块顶部加 `warnings.warn(f"{__name__} moved to ...", DeprecationWarning, stacklevel=2)`
- `concerns/`、`orchestration/`、`providers/`、`schemas/`、`content_types/`、`exceptions/`、`types/`、`utils/`、`observability/` 9 个旧顶层目录降级为"纯 re-export 壳目录"
- 下个 major 版本（v2.0）删除这些壳目录

## 6. 改动影响范围

| 变更 | 文件数 | 风险 |
|------|--------|------|
| `orchestration/` → `memory/` 重命名拆分 | 5 | 低（re-export 兜底） |
| `concerns/` 6 子目录拍平 + 2 合并 | 37 | 中（re-export 兜底） |
| `providers/` 3 子目录上顶层 | 17 | 低（re-export 兜底） |
| `content_types/` + `schemas/` → `models/` 合并 | 15 | 中（block.py 数据/操作合并需谨慎） |
| 4 碎片 → `core/` | 5 | 低 |
| 内部 import 更新（src/ 内互相引用） | ~80 | 中（机械替换，测试兜底） |
| 测试 import 更新（tests/） | ~30 | 低 |
| **合计** | ~79 文件移动 + ~110 import 更新 | |

### block.py 合并特殊处理

`schemas/block.py`（Block dataclass + default_blocks）和 `content_types/working/block.py`（WorkingMemory 操作类）合并到 `models/block.py`：
- 两者都操作 Block，合并后数据模型与操作行为同文件，符合 letta `schemas/memory.py` 惯例
- 合并后检查无循环依赖（WorkingMemory 依赖 Block，同文件内引用）

## 7. 验证标准

1. `ruff check src/ tests/ examples/` 全绿
2. `ruff format --check src/ tests/ examples/` 全绿（注意：不用 `ruff format` 直接改文件，Windows 有清空 bug）
3. `PYTHONPATH=src pytest tests/unit/ tests/e2e/ -q` 基线不退化（686 passed + 22 skipped）
4. 旧路径 re-export 验证：`python -c "from septmuse.concerns.retrieval import HybridRetriever"` 不报错（只 DeprecationWarning）
5. 新路径验证：`python -c "from septmuse.retrieval import HybridRetriever; from septmuse.memory import Memory"` 成功

## 8. 迁移顺序（留给实施计划）

建议分 5 批次，每批后跑测试，降低单次风险：

1. **碎片合并**：`exceptions/types/utils/observability` → `core/`（5 文件，最低风险热身）
2. **数据模型合并**：`schemas + content_types` → `models/`（15 文件，block.py 特殊处理）
3. **orchestration 重命名**：→ `memory/`（5 文件，facade 是核心）
4. **concerns 拍平**：6 子目录上顶层 + 2 合并（37 文件，最大批次）
5. **providers 拆解**：embedders/llms/rerankers 上顶层（17 文件）

每批完成后：ruff check + pytest 基线验证。
