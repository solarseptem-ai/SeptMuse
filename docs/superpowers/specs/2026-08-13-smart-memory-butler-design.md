# SeptMuse 记忆管家 — 设计规格

> 把 SeptMuse 从被动存储升级为主动记忆管家，通过 MCP 协议服务 agent，在记忆质量、演化和多类型上超越 mem0。

## 背景

### 定位

SeptMuse 是**通用 agent 框架的记忆插件**，通过 MCP 协议接入 LangChain/CrewAI/AutoGen 等 agent。agent 只需调 4 个高层 API（remember/recall/improve/forget），不需要管记忆存储细节。

### mem0 的三个核心短板

1. **记忆质量差** — 存一堆噪音，检索时召回无关内容
2. **记忆是静态的** — 不会遗忘、不会演化、不会自我修正矛盾
3. **类型太单一** — 只有 semantic facts，没有情节/程序性/工作记忆区分

### SeptMuse 的差异化特色（已有 + 增强）

| 特色 | mem0 | SeptMuse |
|------|------|----------|
| 记忆类型 | semantic only | working + episodic + semantic + procedural |
| 双时态 | 无 | valid_at / invalid_at（事实有"曾经为真"） |
| 遗忘曲线 | 无 | Ebbinghaus 衰减，记忆自然淡化 |
| 知识图谱 | V3 简单图 | cognify 三元组 → 实体关系 → Zettel 双向链接 |
| 元认知 | 无 | coverage 报告 + rehearsal 复述（agent 知道自己"知道什么/不知道什么"） |
| 编排 API | add/search | remember/recall/improve/forget（全自动管家） |
| 检索配方 | 无 | 8 种预置 recipe |
| **写入质量**（新增） | LLM 抽事实 | 价值评估 + 自动分类 + 语义去重 + 写入时冲突检测 |
| **智能召回**（新增） | 原始列表 | 摘要 + 冲突标注 + 行动建议 + token 裁剪 |
| **后台演化**（新增） | 无 | 周期调度器（蒸馏/冲突解决/图谱生长/遗忘衰减） |
| **LLM 降级**（增强） | 无 LLM 不工作 | 降级为被动存储 + 恢复后补跑 |

## 设计决策

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 1 | 接入方式 | MCP 协议为主 | agent 通过 MCP tools 交互，语言无关 |
| 2 | 记忆管家模式 | 全自动 | 自主决定该不该存、存什么类型、关联什么 |
| 3 | recall 返回 | 智能上下文包 | 摘要 + 记忆列表 + 冲突标注 + 行动建议 |
| 4 | 演化时机 | 混合 | 轻量实时（写入路径），重量后台（调度器） |
| 5 | LLM 降级 | 降级为被动存储 | LLM 不可用时存原文，恢复后补跑 |

## 四阶段交付

### Phase 1：MCP 暴露编排 API

**目标**：agent 通过 MCP 直接调 remember/recall/improve/forget。

**新增 4 个 MCP 工具**：

| 工具名 | 参数 | 返回 | 对应方法 |
|--------|------|------|---------|
| `remember` | `text: str`, `user_id: str`, `agent_id: str?`, `session_id: str?` | `{stored: bool, memory_ids: list, facts_extracted: int, entities_extracted: int, working_memory_updated: bool}` | `ExperimentalMemory.remember()` |
| `recall` | `query: str`, `user_id: str`, `agent_id: str?`, `session_id: str?`, `top_k: int?`, `token_budget: int?` | `{summary: str, memories: list, conflicts: list, injected_prompt: str, recommended_action: str?}` | `ExperimentalMemory.recall()` |
| `improve` | `user_id: str`, `tasks: list[str]?` | `{distilled_rules: int, conflicts_resolved: int, links_grown: int, coverage: dict}` | `ExperimentalMemory.improve()` |
| `forget` | `user_id: str?`, `memory_id: str?` | `{invalidated: int, deleted: int}` | `ExperimentalMemory.forget()` |

**设计要点**：
- 保留现有 18 个底层 MCP 工具（给精细控制用），新增 4 个高层编排工具
- recall 返回的 `summary` 和 `conflicts` 在 Phase 1 为空占位，Phase 2 填充
- improve 的 `tasks` 可选指定子任务（`["distill","resolve_conflicts","grow_graph","analyze_coverage"]`），不传跑全部
- Phase 1 improve 是同步阻塞，Phase 4 改为后台调度

**文件**：`src/septmuse/api/mcp/tools.py`（新增 4 个 `@mcp.tool`）

---

### Phase 2：Smart Recall — 智能上下文包

**目标**：recall 返回的不再是原始列表，是加工好的即用上下文包。

**recall 返回结构**：

```python
{
    "memories": [...],           # 排序后记忆列表（Phase 1 已有）
    "injected_prompt": "<memory>...</memory>",  # 可直接拼 prompt（Phase 1 已有）
    "recommended_action": "DEEPEN_RETRIEVAL",    # 元认知策略（Phase 1 已有）

    # Phase 2 新增
    "summary": "用户计划下周去上海出差，涉及项目X。此前去过上海3次...",
    "conflicts": [
        {
            "topic": "上海出差计划",
            "current": "用户下周要出差去上海",
            "contradicting": "用户上周说不去上海了",
            "contradicting_memory_id": "mem-abc123",
            "suggestion": "建议询问出差日期是否确定"
        }
    ],
}
```

**新增模块**：

1. **RecallSummarizer** (`retrieval/recall_summarizer.py`)
   - LLM 可用：prompt 压缩 N 条记忆为一段摘要，保留关键事实和时间线索
   - LLM 不可用：取 top-3 记忆 memory 字段拼接，截断 200 字

2. **ConflictAnnotator** (`retrieval/conflict_annotator.py`)
   - 只在返回的 top-K 结果内找矛盾（不全库扫描）
   - 规则：相同 subject 不同 object 的 semantic facts → 标记冲突
   - 规则：时间矛盾 → 按时间戳取最新，标记旧的有矛盾
   - LLM 可用：候选冲突对喂 LLM 判定是否真矛盾
   - LLM 不可用：只做规则匹配

3. **集成到 RecallPipeline** — 新增 2 步插在 budget 之后、prompt 之前：
   ```
   compress → search → budget → [summarize] → [annotate_conflicts] → prompt → route → strategy
   ```

**设计要点**：
- summarize + annotate 增加一次 LLM 调用（~2s），但 agent 省去自己理解 N 条记忆的时间
- 冲突不自动解决，只标注。解决留给 improve() 或 agent 显式处理
- LLM 不可用时 summary 是拼接降级版、conflicts 只有规则匹配

---

### Phase 3：Smart Write — 全自动记忆管家写入侧

**目标**：remember 时 SeptMuse 自主决定存什么、怎么分类、该不该存、是否矛盾。

**记忆写入管线（6 步）**：

```
1. 价值评估    → 该不该存？
2. 类型分类    → 存什么类型？
3. 语义去重    → 已有相似记忆吗？
4. 事实抽取    → 抽取结构化事实（现有，增强）
5. 冲突检测    → 和已有记忆矛盾吗？（现有，前移到写入路径）
6. 实体+图谱   → 抽实体、建链接、更新工作记忆（现有）
```

**新增模块**：

1. **ImportanceAssessor** (`capture/importance.py`)
   - 两个维度：信息密度（实体/数字/时间/因果，权重 0.4）+ 持续相关性（计划/偏好/关系 vs 闲聊/状态，权重 0.6）
   - LLM 可用：prompt 打分（"1.0=关键信息，0.5=有用，0.0=噪音"）
   - LLM 不可用：规则——含实体/数字/时间 → 0.7+；含计划/偏好关键词 → 0.6+；其余 → 0.3
   - 阈值 < 0.3 跳过记忆存储，但 raw_log 仍存

2. **MemoryClassifier** (`capture/classifier.py`)
   - LLM 可用：prompt 分类（事实陈述/情节事件/操作步骤/用户偏好）
   - LLM 不可用：关键词决策树（步骤→procedural，时间+动作→episodic+semantic 双存，偏好→semantic，默认→semantic）
   - 一条记忆可同时属于多个类型（双存）

3. **SemanticDeduplicator** (`capture/dedup.py`)
   - 用 query embedding 检索 top-5 已有记忆
   - cosine > 0.92 → 判定重复 → UPDATE 不新增
   - LLM 可用：0.85-0.92 灰色地带 LLM 判定
   - LLM 不可用：只走 0.92 硬阈值

4. **冲突检测前移** — 现有 `ConflictResolver.detect_conflicts()` 从 improve() 批操作前移到写入路径
   - 新事实 (subject, predicate) 和已有冲突 → 标记冲突
   - 不自动解决，在返回值 `result.conflicts` 里返回给 agent

**设计要点**：
- raw_log 永远存（即使价值低跳过记忆存储，审计 + LLM 恢复后补跑）
- 双存：一条文本可同时存为 episodic + semantic
- 冲突不阻塞写入，检测到仍写入但返回冲突信息
- LLM 降级：规则版质量降一档但不阻塞，LLM 恢复后 improve() 可补跑

---

### Phase 4：后台演化调度器 + LLM 恢复补跑

**目标**：agent 不用时 SeptMuse 自己整理记忆。LLM 恢复后自动补跑降级跳过的任务。

**新增模块**：

1. **EvolutionScheduler** (`evolution/scheduler.py`)
   - 后台线程，不阻塞 agent 请求
   - 单线程串行跑（避免 SQLite 写冲突）
   - 4 个演化任务：

   | 任务 | 间隔 | 触发条件 | 对应方法 |
   |------|------|---------|---------|
   | 遗忘衰减 | 1h | 距上次衰减 > 1h | `forgetting.archive_stale()` |
   | 冲突解决 | 5min | 待解决冲突队列非空 | `ConflictResolver.resolve_conflicts()` |
   | 会话蒸馏 | 30min | 自上次蒸馏后新记忆 > 20 条 | `SessionReflector.reflect()` |
   | 图谱生长 | 10min | 待链接队列非空 | `DreamIntegrator.dream()` |

   - MCP server 启动时自动 start，lifespan shutdown 时 stop（等当前任务完成最多 10s）
   - 可配置：`SEPTMUSE_EVOLUTION_INTERVAL`（默认 300s）、`SEPTMUSE_EVOLUTION_ENABLED`（默认 true）

2. **LLMCatchUpQueue** (`evolution/catchup.py`)
   - LLM 不可用时降级跳过的任务进队列（SQLite 表）
   - 字段：`task_type` / `memory_id` / `payload` / `status` / `created_at`
   - task_type：`importance_assessment` / `fact_extraction` / `conflict_judgment` / `summarization` / `classification`
   - 后台调度器每次轮询先 LLM ping，恢复了就消费 queue 补跑
   - 补跑完成后标记 `status=completed`

**设计要点**：
- 不引入新依赖：后台线程用 stdlib `threading.Timer`
- SQLite 安全：单线程串行演化，WAL 模式下读不阻塞
- 每个任务最大执行时间 30s，超时跳过下次再跑
- Phase 1 的 improve() 同步调用仍然可用，调度器是额外的自动触发

## 测试策略

每个 Phase 独立测试：

| Phase | 测试范围 | 测试方式 |
|-------|---------|---------|
| 1 | MCP 工具注册 + 参数校验 + 返回结构 | 单元测试（mock Memory） |
| 2 | RecallSummarizer LLM/降级路径 + ConflictAnnotator 规则/LLM 路径 + Pipeline 集成 | 单元测试 + 集成测试 |
| 3 | ImportanceAssessor 阈值 + MemoryClassifier 决策树 + SemanticDeduplicator cosine + 写入冲突 | 单元测试 + 集成测试 |
| 4 | EvolutionScheduler 调度逻辑 + LLMCatchUpQueue 入队/消费 + 优雅停止 | 单元测试（mock 时间） |

## 非目标

- 不做 Redis 缓存层（P2）
- 不做 k8s 部署（P2）
- 不做多租户分片（P2）
- 不做 OpenTelemetry 分布式追踪（P1）
- 不重写现有存储层（保持 SQLite/PG/MySQL 可切换）
- 不引入 Celery/APScheduler（用 stdlib threading）

## 依赖关系

```
Phase 1 (MCP 暴露) ──→ Phase 2 (Smart Recall) ──→ Phase 3 (Smart Write) ──→ Phase 4 (后台演化)
                           ↑ summary/conflicts        ↑ 价值评估结果入 catch-up     ↑ 消费 catch-up queue
                           ↑ 依赖 recall 返回结构       ↑ 需要冲突检测前移           ↑ 需要 Phase 3 的降级记录
```

Phase 间有依赖但不是硬阻塞——Phase 2 可以在 Phase 1 完成后立即开始，Phase 3 可以和 Phase 2 并行设计但需要 Phase 2 的 recall 返回结构稳定。
