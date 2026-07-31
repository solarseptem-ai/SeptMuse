# Honcho Dreamer 推理整合流程深度解析

> 基于源码 `src/dreamer/` 目录（`orchestrator.py`, `specialists.py`, `surprisal.py`, `dream_scheduler.py`）绘制

---

## 一、Dreamer 定位与核心思想

**Dreamer 是 Honcho 的"夜间整理师"** —— 当 Deriver 生成大量原始 explicit 结论后，Dreamer 负责：

1. **Deduction**（演绎）：从 explicit 推导隐含逻辑（"在 Google 做 SWE" → "有软件工程技能"）
2. **Induction**（归纳）：从多个事实中提炼模式（"周一迟到、周二迟到" → "倾向于迟到"）
3. **Peer Card 维护**：更新稳定的身份快照
4. **Contradiction 检测**：标记矛盾陈述

**核心设计哲学：**
> Deriver 做"快思考"（单次 LLM 调用，事实抽取），Dreamer 做"慢思考"（多轮 Agent 工具循环，深度推理）。

---

## 二、整体架构：Orchestrator + Specialists

```mermaid
flowchart TB
    subgraph Scheduler["DreamScheduler<br/>(调度器)"]
        S1["check_and_schedule_dream()<br/>Deriver 保存后触发"]
        S2["document_threshold 检测<br/>默认 20 条 explicit"]
        S3["min_hours_between_dreams 检测<br/>默认间隔"]
        S4["idle_timeout 延迟<br/>默认 5 分钟"]
        S5["enqueue_dream()<br/>入队 queue_items"]
    end

    subgraph Orchestrator["Orchestrator<br/>(指挥者)"]
        O1["run_dream()<br/>Worker 消费队列"]
        O2["可选: Surprisal 采样<br/>筛选高信息量结论"]
        O3["DeductionSpecialist<br/>演绎专家"]
        O4["InductionSpecialist<br/>归纳专家"]
        O5["CardRefreshSpecialist<br/>身份卡维护"]
        O6["run_card_refresh_dream()<br/>轻量级卡片刷新"]
    end

    subgraph Specialists["Specialists<br/>(工具循环 Agent)"]
        SP1["Phase 1: Discovery<br/>get_recent_observations<br/>search_memory<br/>search_messages"]
        SP2["Phase 2: Action<br/>create_observations_*<br/>delete_observations<br/>update_peer_card"]
        SP3["工具循环<br/>最多 10-15 轮"]
    end

    subgraph Storage["PostgreSQL / pgvector"]
        D1["documents<br/>(explicit)"]
        D2["documents<br/>(deductive)"]
        D3["documents<br/>(inductive)"]
        D4["documents<br/>(contradiction)"]
        D5["reasoning_trees<br/>(premise → conclusion 链接)"]
        D6["peer_cards<br/>(身份快照)"]
    end

    S1 --> S2
    S2 -->|达标| S3
    S3 -->|间隔足够| S4
    S4 -->|延迟后| S5
    S5 -->|Worker 消费| O1

    O1 -->|surprisal.enabled=true| O2
    O2 --> O3
    O3 --> O4
    O1 -->|dream_type=OMNI| O3

    O1 -->|dream_type=CARD_REFRESH| O6
    O6 --> O5

    O3 --> SP1
    O4 --> SP1
    O5 --> SP1
    SP1 --> SP3
    SP3 --> SP2
    SP2 --> SP3
    SP3 -->|完成或达到迭代上限| O1

    SP2 -->|create_observations_deductive| D2
    SP2 -->|create_observations_inductive| D3
    SP2 -->|create_observations_contradiction| D4
    SP2 -->|delete_observations| D1
    SP2 -->|update_peer_card| D6
    D2 --> D5
    D3 --> D5
    D4 --> D5

    style Orchestrator fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Specialists fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 三、Surprisal（惊讶度）采样机制

### 3.1 为什么需要 Surprisal？

Deriver 源源不断地产生 explicit 结论（可能数百条），Dreamer 不可能每次都处理全部。**Surprisal 采样选出"最出人意料"的结论**，让 Dreamer 聚焦高价值信息。

### 3.2 Surprisal 计算流程

```mermaid
flowchart LR
    subgraph Fetch["1. 获取观测"]
        F1["从 documents 表取样本"]
        F2["策略: recent / random / all"]
        F3["过滤: level ∈ [explicit, deductive]"]
    end

    subgraph Compute["2. 计算惊讶度"]
        C1["提取 embeddings"]
        C2["构建 SurprisalTree<br/>(BallTree / KDTree)"]
        C3["对每个观测:<br/>surprisal = 到第 K 近邻的平均距离"]
        C4["Min-Max 归一化到 [0,1]"]
    end

    subgraph Filter["3. 筛选"]
        FI1["按 surprisal 降序"]
        FI2["取 top_percent<br/>默认 10%"]
        FI3["转为 exploration_hints"]
    end

    F1 --> F2 --> F3 --> C1
    C1 --> C2 --> C3 --> C4
    C4 --> FI1 --> FI2 --> FI3
```

### 3.3 几何直觉

```
高维向量空间中:

  ● 普通结论（邻居密集）          ● 出人意料结论（邻居稀疏）
  ↓ surprisal 低                  ↓ surprisal 高

     ● ● ●
    ● ●A● ●        A 的 5-NN 距离小        ●
     ● ● ●                              ●   B
                                        ●
                                          ●
                                        B 的 5-NN 距离大 → 高 surprisal
```

**数学定义：**
```python
surprisal(embedding) = mean_distance_to_k_nearest_neighbors(embedding, k=TREE_K)
```

### 3.4 采样策略配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SAMPLING_STRATEGY` | `recent` | 采样策略：`recent` / `random` / `all` |
| `SAMPLE_SIZE` | 500 | 最大采样数 |
| `TREE_TYPE` | `balltree` | 树结构：`balltree` / `kdtree` |
| `TREE_K` | 5 | K 近邻数 |
| `TOP_PERCENT_SURPRISAL` | 0.1 | 取 top 10% |
| `INCLUDE_LEVELS` | `["explicit", "deductive"]` | 参与计算的结论层级 |

---

## 四、Specialist 工具循环详解

### 4.1 三种 Specialist 对比

| 维度 | **DeductionSpecialist** | **InductionSpecialist** | **CardRefreshSpecialist** |
|------|------------------------|------------------------|--------------------------|
| **目标** | 逻辑推导 + 清理过时结论 | 模式归纳 + 行为洞察 | 维护 Peer Card 身份快照 |
| **可写结论类型** | deductive + contradiction | inductive | 无（只更新 peer_card） |
| **可删结论** | ✅ 可以删除过时 explicit | ❌ 不删除 | ❌ 不删除 |
| **可更新 Peer Card** | ✅ | ❌ | ✅ |
| **最大迭代** | 12 | 10 | 6 |
| **模型配置** | `DREAM.DEDUCTION_MODEL_CONFIG` | `DREAM.INDUCTION_MODEL_CONFIG` | 复用 DEDUCTION |

### 4.2 DeductionSpecialist 的两阶段工作流

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Discovery（探索）"]
        D1["get_recent_observations<br/>看最近学了什么"]
        D2["search_memory<br/>向量搜索特定话题"]
        D3["search_messages<br/>看原始对话内容"]
    end

    subgraph Phase2["Phase 2: Action（行动）"]
        A1{"发现知识更新?"}
        A2["create_observations_deductive<br/>创建演绎结论"]
        A3["delete_observations<br/>删除过时 explicit"]
        A4{"发现矛盾?"}
        A5["create_observations_contradiction<br/>标记矛盾"]
        A6["update_peer_card<br/>更新身份卡"]
    end

    Phase1 --> Phase2
    A1 -->|是| A2
    A1 -->|是| A3
    A4 -->|是| A5
    Phase2 --> A6
```

**Deduction 的典型产出：**

```python
# 输入 explicit:
#   "alice works as SWE at Google"
#   "alice has 5 years of experience"

# 输出 deductive:
DeductiveObservation(
    conclusion="alice has software engineering skills",
    premises=["alice works as SWE at Google", "alice has 5 years of experience"],
    source_ids=["doc_123", "doc_124"],
)

# 知识更新场景:
#   旧 explicit: "alice lives in NYC"
#   新 explicit: "alice moved to SF last month"
# → 创建 deductive: "alice lives in SF"
# → 删除旧 explicit (标记过时)
```

### 4.3 InductionSpecialist 的两阶段工作流

```mermaid
flowchart TB
    subgraph IP1["Phase 1: Discovery"]
        I1["get_recent_observations<br/> explicit + deductive"]
        I2["search_memory<br/>跨话题搜索"]
        I3["search_messages<br/>看原始对话"]
    end

    subgraph IP2["Phase 2: Action"]
        I4{"发现跨观测模式?"}
        I5["create_observations_inductive<br/>创建归纳结论"]
    end

    IP1 --> IP2
    I4 -->|是| I5
```

**Induction 的典型产出：**

```python
# 输入多个 explicit:
#   "alice rescheduled Monday's meeting"
#   "alice rescheduled Tuesday's meeting"
#   "alice said she was stressed about deadlines"

# 输出 inductive:
InductiveObservation(
    conclusion="alice tends to reschedule meetings when under deadline pressure",
    sources=["alice rescheduled Monday...", "alice rescheduled Tuesday...", "alice said she was stressed..."],
    source_ids=["doc_1", "doc_2", "doc_3"],
    pattern_type="behavior",
    confidence="medium",  # 3-4 条证据
)
```

### 4.4 工具集对比

```mermaid
flowchart LR
    subgraph DeductionTools["DeductionSpecialist Tools"]
        DT1["get_recent_observations"]
        DT2["search_memory"]
        DT3["search_messages"]
        DT4["create_observations_deductive"]
        DT5["delete_observations"]
        DT6["update_peer_card"]
    end

    subgraph InductionTools["InductionSpecialist Tools"]
        IT1["get_recent_observations"]
        IT2["search_memory"]
        IT3["search_messages"]
        IT4["create_observations_inductive"]
    end

    subgraph CardRefreshTools["CardRefreshSpecialist Tools"]
        CT1["get_recent_observations"]
        CT2["search_memory"]
        CT3["update_peer_card"]
    end
```

---

## 五、完整的 Dream 生命周期时序图

```mermaid
sequenceDiagram
    autonumber
    participant Deriver as Deriver Worker
    participant Scheduler as DreamScheduler
    participant Queue as queue_items
    participant Worker as Deriver Worker<br/>(Consumer)
    participant Orchestrator as run_dream()
    participant Surprisal as Surprisal采样
    participant Deduction as DeductionSpecialist
    participant Induction as InductionSpecialist
    participant DB as PostgreSQL

    %% 触发阶段
    Deriver->>DB: save_representation()<br/>写入 explicit 结论
    Deriver->>Scheduler: check_and_schedule_dream()
    Scheduler->>DB: count explicit documents
    DB-->>Scheduler: current = 25, last = 5
    Scheduler->>Scheduler: 25-5=20 ≥ threshold(20) ✓
    Scheduler->>Scheduler: min_hours 检查 ✓
    Scheduler->>Scheduler: 无 pending dream ✓
    Scheduler->>Scheduler: asyncio.create_task(<br/>delay=IDLE_TIMEOUT_MINUTES)
    Note over Scheduler: 等待 5 分钟（攒更多结论）

    %% 入队阶段
    Scheduler->>Queue: enqueue_dream()<br/>work_unit_key="dream:app:::alice"

    %% 消费阶段
    Worker->>Queue: 轮询发现 dream task
    Worker->>Worker: claim work_unit
    Worker->>Orchestrator: process_dream()<br/>dream_type=OMNI

    %% Surprisal 阶段
    Orchestrator->>Surprisal: sample_observations_with_surprisal()
    Surprisal->>DB: fetch 500 documents
    DB-->>Surprisal: explicit list
    Surprisal->>Surprisal: build BallTree from embeddings
    Surprisal->>Surprisal: compute k-NN distances
    Surprisal->>Surprisal: normalize + top 10%
    Surprisal-->>Orchestrator: high_surprisal_obs: [obs_A, obs_B, ...]
    Orchestrator->>Orchestrator: hints = [obs_A.content, obs_B.content, ...]

    %% Deduction 阶段
    Orchestrator->>Deduction: run(observer, observed, hints=hints)
    loop Tool Loop (max 12 iterations)
        Deduction->>DB: get_recent_observations()
        DB-->>Deduction: recent explicit
        Deduction->>DB: search_memory("career")
        DB-->>Deduction: relevant docs
        Deduction->>Deduction: LLM 推理 → 调用工具
        Deduction->>DB: create_observations_deductive()
        Deduction->>DB: delete_observations(outdated_ids)
        Deduction->>DB: update_peer_card()
    end
    Deduction-->>Orchestrator: SpecialistResult<br/>created: 5, deleted: 2, card_updated: true

    %% Induction 阶段
    Orchestrator->>Induction: run(observer, observed, hints=hints)
    loop Tool Loop (max 10 iterations)
        Induction->>DB: get_recent_observations()
        DB-->>Induction: explicit + deductive
        Induction->>DB: search_memory("behavior")
        DB-->>Induction: pattern candidates
        Induction->>Induction: LLM 推理 → 调用工具
        Induction->>DB: create_observations_inductive()
    end
    Induction-->>Orchestrator: SpecialistResult<br/>created: 3

    %% 收尾阶段
    Orchestrator->>DB: UPDATE collection.dream_meta<br/>last_dream_at = now<br/>last_dream_document_count = current_explicit
    Orchestrator->>Orchestrator: emit DreamRunEvent
```

---

## 六、推理树（Reasoning Trees）

### 6.1 数据模型

```mermaid
flowchart LR
    subgraph Explicit["Explicit Layer"]
        E1["doc_101: alice is 25"]
        E2["doc_102: alice works at Google"]
        E3["doc_103: alice has a dog"]
    end

    subgraph Deductive["Deductive Layer"]
        D1["doc_201: alice is employed in tech"]
        D2["doc_202: alice has dependents<br/>(pet = dependent)"]
    end

    subgraph Inductive["Inductive Layer"]
        I1["doc_301: alice is career-focused<br/>and family-oriented"]
    end

    E1 -->|premise| D1
    E2 -->|premise| D1
    E2 -->|premise| I1
    E3 -->|premise| D2
    E3 -->|premise| I1
    D1 -->|premise| I1
    D2 -->|premise| I1
```

### 6.2 reasoning_trees 表结构

```sql
-- 推理树记录每条结论的来源
CREATE TABLE reasoning_trees (
    id TEXT PRIMARY KEY,
    workspace_name TEXT NOT NULL,
    observer TEXT NOT NULL,
    observed TEXT NOT NULL,
    conclusion_document_id TEXT NOT NULL,    -- 当前结论 doc_id
    premise_document_id TEXT NOT NULL,        -- 前提 doc_id
    created_at TIMESTAMP DEFAULT NOW()
);
```

**用途：**
- `get_reasoning_chain(doc_id)` — 回溯一条结论的完整推理链
- 可解释性：Dialectic 回答时可以展示"为什么我知道这个"
- 置信度评估：链条越长/分支越多，结论越可靠

---

## 七、Peer Card 身份快照

### 7.1 四条严格格式规则

Peer Card 是**稳定身份标记的快照**，不是动态观察：

| 前缀 | 类型 | 示例 |
|------|------|------|
| `IDENTITY:` | 标识信息 | `IDENTITY: Name: Alice` |
| `ATTRIBUTE:` | 稳定属性 | `ATTRIBUTE: Location: NYC` |
| `RELATIONSHIP:` | 持久关系 | `RELATIONSHIP: Spouse: Bob` |
| `INSTRUCTION:` | 交互规则 | `INSTRUCTION: Call me Vee` |

### 7.2 稳定性规则

> **"如果值在六个月内可能变化（没有明确公告），就不属于 Peer Card。"**

```mermaid
flowchart TB
    A["候选信息"] --> B{"六个月内会变化?"}
    B -->|是| C["放入 observations<br/>(inductive/explicit)"]
    B -->|否| D["放入 Peer Card<br/>(稳定身份)"]
```

### 7.3 Card Refresh 的两种模式

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **Refresh** | 默认 | 读取现有 card + 新观测 → 增量更新 |
| **Rebuild** | 大量删除后 | 不读现有 card → 从观测完全重建 |

---

## 八、Dream 调度策略

### 8.1 触发条件（与门逻辑）

```mermaid
flowchart LR
    A["Deriver 写入 explicit"] --> B{"条件检查"}
    B --> C1["DREAM.ENABLED=true?"]
    C1 -->|是| C2["explicit_count - last_count ≥ threshold?"]
    C2 -->|是| C3["hours_since_last ≥ MIN_HOURS?"]
    C3 -->|是| C4["无 pending dream?"]
    C4 -->|是| D["schedule_dream()<br/>延迟 IDLE_TIMEOUT 分钟"]
```

### 8.2 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DREAM.ENABLED` | true | 总开关 |
| `DOCUMENT_THRESHOLD` | 20 | 自上次 dream 新增的 explicit 数阈值 |
| `MIN_HOURS_BETWEEN_DREAMS` | 1 | 两次 dream 最小间隔 |
| `IDLE_TIMEOUT_MINUTES` | 5 | 达标后延迟多久执行（攒更多结论） |
| `ENABLED_TYPES` | `["omni", "card_refresh"]` | 启用的 dream 类型 |

### 8.3 防重复机制

```mermaid
flowchart TB
    subgraph Guard["Guard Pair（互斥锁）"]
        G1["last_dream_at<br/>上次执行时间"]
        G2["last_dream_document_count<br/>上次执行时的 explicit 数"]
    end

    subgraph ScheduleCheck["调度检查"]
        S1["当前 explicit = 120"]
        S2["last_dream_document_count = 100"]
        S3["差值 = 20 ≥ threshold"]
        S4["pending_exists = false"]
    end

    subgraph Execution["执行后更新"]
        E1["last_dream_at = now()"]
        E1 --> E2["last_dream_document_count = 120"]
    end

    G1 --> ScheduleCheck
    G2 --> ScheduleCheck
    ScheduleCheck -->|通过| Execution
```

---

## 九、与 Mem0 的架构对比

| 维度 | **Honcho Dreamer** | **Mem0** |
|------|-------------------|----------|
| **推理时机** | 异步批处理（攒够阈值后触发） | 同步（add() 时立即处理） |
| **推理 Agent** | 多 Specialist 工具循环（Deduction + Induction） | 单次 LLM 调用（fact extraction） |
| **结论分层** | explicit → deductive → inductive | 单层 facts |
| **推理链** | ✅ 显式追踪（reasoning_trees） | ❌ 隐式（linked_memory_ids） |
| **用户画像** | ✅ Peer Card（稳定身份快照） | ❌ 无 |
| **矛盾处理** | ✅ 显式 contradiction 结论 | ❌ 依赖后续 add 覆盖 |
| **模式发现** | ✅ InductionSpecialist 自动归纳 | ❌ 无自动归纳 |
| **惊喜度驱动** | ✅ Surprisal 采样聚焦高价值 | ❌ 无 |
| **成本** | 高（多轮 Agent + 延迟处理） | 低（单次调用 + 立即返回） |

---

## 十、对 SeptMuse 的启示

### 10.1 可借鉴的架构模式

| 模式 | SeptMuse 现状 | 建议 |
|------|--------------|------|
| **多层推理** | `TripletExtractor` 抽三元组 | 增加 Deduction/Induction 分层，显式记录推理链 |
| **Surprisal 采样** | 无 | 用 K-NN 距离筛选高价值记忆，减少 LLM 调用量 |
| **Peer Card** | 无 | 为每个 user 维护稳定身份快照，注入 Dialectic prompt |
| **异步 Dream** | 全同步 | 可选后台 worker，攒批后深度推理 |
| **工具循环 Agent** | 无 | Dialectic/Reflect 可用工具循环替代单次调用 |

### 10.2 实现优先级建议

```
Phase 1（轻量版）:
  └─ 在 add() 后异步触发一次"轻量 Deduction"
  └─ 用简单的规则（关键词匹配）做 implicit implication

Phase 2（完整版）:
  └─ 引入 Worker + Queue 架构
  └─ 实现 DeductionSpecialist（工具循环）
  └─ 引入 reasoning_trees 表

Phase 3（高级版）:
  └─ InductionSpecialist（模式归纳）
  └─ Surprisal 采样
  └─ Peer Card 维护
```

---

*报告完*
