# SeptMuse Agent 记忆系统架构设计

> 模块定位：solarseptem-ai 平台的第 4 个子系统 SeptMuse（缪斯/记忆），为 SolAgent 及其它子系统提供统一的 agent 记忆能力。
> 技术栈对齐：FastAPI + SQLModel（后端） / Next.js（前端） / 可选服务模块（model_gateway / mcp_market / agent_runner）。
> 调研基础：14 家主流开源 agent 记忆实现（mem0 / Letta / ReMe / Cognee / Zep-Graphiti / LangMem / A-MEM / MemOS / Basic Memory / Cass / Agent Memory / Agno / Hermes / Graphiti）。

---

## 1. 设计目标与原则

### 1.1 设计目标

| 目标 | 衡量标准 |
|------|---------|
| 全形态覆盖 | 文本/向量/图/文件/激活(KV-cache)/参数化(LoRA) 六种存储形态皆可插拔 |
| 认知分层清晰 | 工作/情节/语义/程序 四类内容严格分离，不混轴 |
| 跨 agent 共享 | 同一用户记忆可在 SolAgent 多 agent / 多平台间复用 |
| 可审计可治理 | 记忆人可读、写操作可审批、token 预算可控、规则可退化 |
| 三项创新增量 | 因果链 / 遗忘曲线 / 元认知自描述 —— 填补 14 家开源均未覆盖的空白 |
| 最小依赖 | 优先复用 opensource/ 下已克隆的 mem0 / letta / ReMe，减少新依赖 |

### 1.2 设计原则

1. **三维正交**：内容类型 / 存储形态 / 横切关注点 三个平面分离，独立演进，禁止混轴。
2. **多形态共存**：同一份记忆可同时写入多个存储形态（如语义事实 = 图三元组 + Markdown + 向量），由"源同步器"保证一致。
3. **认知科学对齐**：内容类型严格遵循 Atkinson-Shiffrin + Tulving 层级（工作记忆 / 长时记忆[情节·语义·程序]，身份归入语义子类，激活归入工作记忆神经层）。
4. **借鉴优先于自研**：存储后端、抽取流水线、向量/图检索等成熟能力直接复用开源；仅"创新空白"和"统一编排层"自研。
5. **渐进落地**：先跑通 L1 工作 + L3b 语义最小闭环，再逐层叠加，每层可独立验证。
6. **零 LLM 可用**：所有功能在无 LLM 时有降级方案，保证零配置零依赖可用。降级路径：LLM 抽取 → regex/规则模板；LLM 推理 → 图遍历/统计；LLM 自述 → 纯数字报告。§2.1 组合矩阵每个格子标注 LLM 依赖度（必需/可降级/不需要），§7.2 自研表每项补降级方案。无 LLM 时系统退化为 verbatim 原文存储 + 向量检索 + 规则演化，不崩溃。

---

## 2. 三维正交架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  平面 A：内容类型（what is remembered）—— 主轴                   │
│                                                                  │
│  感觉记忆    (输入 token, <1s, 不持久化)                          │
│  工作记忆    (context 内, 零检索)                                 │
│    └─ 激活   (KV 张量, 神经层)                                    │
│  长时记忆    (跨会话, 需召回)                                     │
│    ├─ 情节   (带时间锚点的事件/经历)                              │
│    ├─ 语义   (事实/偏好/关系)                                     │
│    │   └─ 身份 (人设/自我, 语义子类)                             │
│    └─ 程序   (how-to/skill/规则)                                 │
└─────────────────────────────────────────────────────────────────┘
                              ×
┌─────────────────────────────────────────────────────────────────┐
│  平面 B：存储形态（how stored）—— 每个类型可多形态共存            │
│                                                                   │
│  block | 向量 | 图 | 文件 | 激活 | 参数化                          │
└─────────────────────────────────────────────────────────────────┘
                              ×
┌─────────────────────────────────────────────────────────────────┐
│  平面 C：横切关注点（how managed）—— 贯穿所有类型/形态            │
│                                                                   │
│  捕获 → 检索策略(含元认知路由) → 治理 → 演化 → 共享 → 源同步      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 组合矩阵

|              │ block | 向量 |  图  | 文件 | 激活 | 参数化 |
|--------------|:-----:|:----:|:----:|:----:|:----:|:-----:|
| 工作记忆      |  ✅   |  —   |  —   |  —   |  ✅  |  —    |
| 情节（长时）  |  —    |  ✅  |  ✅  |  ✅  |  —   |  —    |
| 语义（长时）  |  (✅) |  ✅  |  ✅  |  ✅  |  —   |  —    |
| 程序（长时）  |  ✅   |  ✅  |  —   |  ✅  |  —   |  ✅   |

> 每个格子 = (内容类型, 存储形态) 组合，叠加平面 C 的 6 个横切插槽。`✅` 表示该组合成立，`(✅)` 表示身份子类借用 block。

### 2.2 LLM 依赖度分级（零 LLM 可用原则的落地）

每个格子按 LLM 依赖度分三级：

| 级别 | 含义 | 无 LLM 时行为 |
|------|------|---------------|
| **不需要** | 纯数据结构/存储/检索，无语义理解 | 正常工作 |
| **可降级** | LLM 是首选，但有规则/统计/regex 替代 | 降级为规则模式，精度下降但不崩溃 |
| **必需** | 必须用 LLM 抽取/推理/生成 | 跳过该格子，等 LLM 可用时再触发 |

按格子标注：

| 格子 | LLM 依赖 | 降级方案 |
|------|:--------:|----------|
| 工作 × block | 不需要 | — |
| 工作 × 激活 | 不需要 | — |
| 情节 × 向量 | 不需要 | — |
| 情节 × 图 | 可降级 | regex 实体抽取 + 相邻实体规则边（替代 LLM 联合抽取） |
| 情节 × 文件 | 不需要 | — |
| 语义 × block | 不需要 | — |
| 语义 × 向量 | 可降级 | 用户显式 add_fact（替代 LLM 自动抽三元组） |
| 语义 × 图 | 可降级 | 同上，三元组由用户/规则提供 |
| 语义 × 文件 | 不需要 | — |
| 程序 × block | 不需要 | — |
| 程序 × 向量 | 必需 | reflect 蒸馏依赖 LLM；无 LLM 时程序库为空，仅用户显式 add_rule |
| 程序 × 文件 | 不需要 | — |
| 程序 × 参数化 | 不需要 | 训练由调用方用 transformers Trainer 完成，非 LLM 推理 |

**横切关注点 LLM 依赖**：

| 横切 | LLM 依赖 | 降级方案 |
|------|:--------:|----------|
| 捕获（去重/脱敏） | 不需要 | — |
| 检索（向量+BM25+entity boost） | 不需要 | HashEmbedder 离线 + regex 实体 |
| 检索（元认知 L0 路由） | 不需要 | embedding 相似度路由 |
| 治理（token 预算/审批/脱敏） | 不需要 | — |
| 演化（链接生长） | 可降级 | embedding 相似度找关联（不抽语义关系） |
| 演化（Dream integrate） | 可降级 | 规则驱动（高频共现 → CORROBORATE） |
| 演化（reflect 蒸馏） | 必需 | 无 LLM 跳过，无 lessons 产出 |
| 演化（冲突解决） | 可降级 | 精确归一化 + difflib 模糊匹配（LLM 是兜底） |
| 演化（cognify 抽实体边） | 可降级 | regex 实体 + 相邻规则边 |
| 共享（RBAC） | 不需要 | — |
| 元认知 L1 覆盖报告 | 可降级 | 纯统计报告（"语义库 50 条,程序库 3 条"），无自然语言自述 |
| 元认知 L2 策略自调 | 可降级 | 规则驱动（覆盖度 < 阈值 → 触发澄清提问） |
| 因果链抽取 | 必需 | 无 LLM 跳过自动抽取；仅用户显式标注或关键词规则（"导致"/"因为"） |
| 反事实推理 | 必需 | 纯图遍历返回路径（无 LLM 推理"若 X 未发生 Y 是否仍发生"） |
| 遗忘曲线（decay/rehearse） | 不需要 | 纯数学 exp(-t/S) |
| 源同步器 | 不需要 | hash 比对漂移检测 |

**零 LLM 模式下系统形态**：verbatim 原文存储 + HashEmbedder 向量检索 + regex 实体 + 规则链接生长 + 纯统计元认知 + 遗忘曲线衰减。语义事实库为空（除非用户显式 add_fact），程序库为空（除非用户显式 add_rule）。系统退化为"高保真日志 + 检索 + 衰减"，不崩溃。

---

## 3. 平面 A：内容类型详细设计

### 3.1 工作记忆（Working Memory）

**定义**：在 LLM context window 内，零检索即可见的记忆。Agent 可用工具自编辑。

#### 3.1.1 文本块（Block）

借鉴 **Letta Core Block**。

```python
class Block(SQLModel, table=True):
    id: str                      # 块唯一 ID
    label: str                   # 段标签, 如 "human" / "persona" / "task"
    value: str                   # 当前内容
    limit: int                   # 字符上限 (治理: 防止 context 溢出)
    tags: list[str] = []         # 关联标签
    agent_id: str                # 归属 agent

    # 自编辑工具 (暴露给 LLM)
    @tool
    def core_memory_append(self, label: str, content: str) -> None: ...

    @tool
    def core_memory_replace(self, label: str, old: str, new: str) -> None:
        # old 不在 block 内则 raise, 保证精确编辑
        ...

    # 编译为 XML 注入 system prompt
    def compile_to_xml(self) -> str: ...
```

**编译输出**（注入 system prompt）：
```xml
<memory>
  <block label="human">Name: Timber. Occupation: ...</block>
  <block label="persona">I am a self-improving agent...</block>
</memory>
```

**治理增量**（借鉴 Hermes）：在 Block 上加 `char_limit` 硬上限，超限触发压缩或驱逐到长时记忆。

#### 3.1.2 激活记忆（KV-Cache）

借鉴 **MemOS KVCacheMemory**。

**仅当使用自托管模型（HuggingFace 后端）时启用**；API/闭源模型跳过此格。

```python
class KVCacheMemory:
    def extract(self, prompt: list[dict]) -> KVCacheItem:
        # 前向计算得到 DynamicCache (K/V 张量)
        ...

    def add(self, items: list[KVCacheItem]) -> None: ...
    def get_cache(self, ids: list[str]) -> DynamicCache:
        # merge 多个 cache, 注入 attention 跳过 prefill
        ...
    def dump(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**用途**：固定背景/FAQ/会话历史复用，降低 TTFT。

### 3.2 长时记忆（Long-Term Memory）

跨会话持久，需检索召回。按内容类型分四子层。

#### 3.2.1 情节记忆（Episodic）

带时间锚点的事件/经历。三种互补实现，各司其职：

| 子实现 | 借鉴 | 职责 | Schema |
|--------|------|------|--------|
| 时序事件 | Zep/Graphiti Episode | 事实事件 + 时间演化 | `Episode(content, type, reference_time)` |
| 推理经验 | LangMem Episode | 成功交互的推理链 | `Episode(observation, thoughts, action, result)` |
| 原始日志 | Cass Episodic | 高保真 raw session log | session transcript (JSONL) |

```python
class EpisodicEvent(SQLModel, table=True):
    id: str
    content: str                 # 事件内容
    event_type: str              # "fact" | "reasoning" | "raw_log"
    reference_time: datetime    # 时间锚点 (时序检索依据)
    agent_id: str
    session_id: str | None
    # 推理经验专用字段 (event_type="reasoning")
    observation: str | None
    thoughts: str | None
    action: str | None
    result: str | None
```

#### 3.2.2 语义记忆（Semantic）

事实/偏好/关系。身份归入子类（打 `identity` 标签，不单列层）。

```python
class SemanticFact(SQLModel, table=True):
    id: str
    subject: str                 # 三元组
    predicate: str
    object: str
    context: str | None          # 上下文限定
    namespace: tuple             # 多租户 ("memories", org_id, user_id)
    confidence: float = 1.0     # 置信度 (创新: 区分事实/推断)
    provenance: str              # 来源溯源 (user|inferred|tool|observed)
    tags: list[str] = []         # identity/role/preference/...
    embedding: bytes             # 向量 (平面B共存)
```

**抽取流水线**借鉴 Cognee cognify：
```
classify → extract_chunks → extract_graph_from_data (LLM 抽实体/关系)
        → summarize_text → add_data_points (图库+向量库双写)
```

**检索**借鉴 mem0 多信号融合：语义相似 + 关键词 + 实体链接 + 时间相关性。

#### 3.2.3 程序记忆（Procedural）

how-to / skill / 规则。借鉴 **Cass Playbook**（helpful/harmful 计数 + 溯源 + 废弃）。

```python
class ProceduralRule(SQLModel, table=True):
    id: str
    rule: str                    # 规则内容
    namespace: tuple
    helpful_count: int = 0       # 该规则带来正面结果次数
    harmful_count: int = 0       # 带来负面结果次数
    source_tracing: str          # 溯源到具体 episodic session
    deprecated: bool = False     # 规则退化标记
    confidence: float           # = helpful / (helpful+harmful)
```

**退化机制**：当 `harmful_count > helpful_count` 或显式 deprecate，规则不再注入 context。从 working(Diary) 经 `reflect + curate` 升华而来。

### 3.3 身份记忆

归入 L3.2.2 语义记忆，打 `identity` 标签。同时在工作记忆里以 `persona` block 镜像（agent 可自编辑）。

---

## 4. 平面 B：存储形态详细设计

| 形态 | 实现选型 | 借鉴 | 备注 |
|------|---------|------|------|
| block | 内存 + SQLModel 持久化 | Letta Block | context 内 XML 编译 |
| 向量 | pgvector (Postgres 扩展) | mem0 / LangMem Store | 复用 solarseptem 后端 Postgres |
| 图 | Neo4j 或 Apache AGE (Postgres 图扩展) | Graphiti / Cognee | 优先 AGE 减少依赖 |
| 文件 | Markdown + frontmatter + wikilinks, SQLite 索引 | Basic Memory | 人可读, 双向同步 |
| 激活 | KV 张量 (DynamicCache) | MemOS KVCacheMemory | 仅自托管模型 |
| 参数化 | LoRA / adapter | MemOS Parametric | 仅高吞吐自托管场景 |

### 4.1 源同步器（Source Synchronizer）—— 多形态共存一致性

**自研**。当一份记忆需多形态共存时（如语义事实同时写图三元组 + Markdown + 向量），由源同步器保证一致。

```python
class SourceSynchronizer:
    """多形态记忆的最终一致同步器"""
    def write(self, fact: SemanticFact, targets: list[Form]) -> None:
        # 并行写 图/文件/向量, 任一失败则记录补偿任务
        ...

    def reconcile(self, fact_id: str) -> None:
        # 检测形态间漂移, 以 graph 为权威源重同步文件/向量
        ...
```

**权威源优先级**：图 > 文件 > 向量（图最结构化，文件人可纠错，向量易漂移）。

---

## 5. 平面 C：横切关注点详细设计

### 5.1 捕获方式

| 方式 | 借鉴 | 适用 |
|------|------|------|
| 被动 add | mem0 / Cognee | 应用显式调用 |
| agent 自治工具 | Letta core_memory_* / LangMem manage tool | agent 自管 |
| 隐式 hook | Agent Memory PostToolUse | 编码 agent 零侵入 |
| 自演化 | A-MEM Zettelkasten | add 时自动找链接 |

**hook 流水线**（借鉴 Agent Memory）：
```
PostToolUse → SHA-256 去重(5min 窗) → 隐私过滤(脱敏 secrets/keys)
           → 存 raw → LLM 压缩(事实+概念+叙述) → 向量化 → BM25+向量双索引
SessionEnd → 摘要 + 图抽取 + slot 反思
SessionStart → 加载 project profile + 混合检索 + token 预算(默认 2000) 注入
```

### 5.2 检索策略（含元认知路由）

借鉴 ReMe 渐进式三层 + Agent Memory 混合。

```python
class RetrievalStrategy:
    def retrieve(self, query: str, namespace: tuple) -> list[Memory]:
        # Layer 0: 元认知路由 (ReMe meta)
        meta = read_meta_memory()         # 列出所有可用类型/目标
        targets = meta_route(query, meta) # 决定查哪些命名空间

        # Layer 1+2: 混合检索 (Agent Memory)
        results = hybrid_search(query, targets)
        # = BM25 + 向量 + 图扩展 三路融合

        # Layer 3: 按需加载完整历史 (ReMe)
        if needs_full_context(results):
            results += read_history(results[0].ref_memory_id)
        return results
```

**元认知路由**（见 §6.3）决定查哪个命名空间，避免全量扫描。

### 5.3 治理

| 机制 | 借鉴 | 实现位置 |
|------|------|---------|
| token 预算 | Agent Memory (2000) / Hermes (800) | 检索注入前裁剪 |
| 写审批 | Hermes write_approval | Block/规则写入前拦截 |
| 隐私脱敏 | Agent Memory 隐私过滤 | hook 流水线 |
| 规则退化 | Cass helpful/harmful + deprecation | 程序记忆 |
| 置信度加权 | Cass source tracing + 本设计 confidence | 检索排序 |

### 5.4 演化

| 机制 | 借鉴 | 触发 |
|------|------|------|
| 链接生长 | A-MEM Zettelkasten | add/update 时自动找语义关系 |
| 反思升华 | Cass reflect + curate | SessionEnd: Diary → Playbook |
| Dream 整合 | ReMe Dream | 空闲期批量建立 wikilink |
| 图谱增量 | Graphiti 实时增量 | 事实变化时无需批重算 |

### 5.5 共享

借鉴 Agno user_id 跨 agent 共享 + Cass 统一 episodic。

```python
# 多 agent 连同一 Postgres, 用同一 user_id 即可互读记忆
db = SqliteDb / PostgresDb
agent_a = Agent(db=db, user_id="alice")  # chat agent 学的
agent_b = Agent(db=db, user_id="alice")  # research agent 能用
```

**跨 agent 范围**：同 user_id 语义/程序记忆可共享；情节 raw log 统一入 episodic 池（Cass 模式）。

### 5.6 元认知路由（归此横切关注点，状态持久化为语义记忆）

详见 §6.3。

---

## 6. 三个创新空白（自研）

14 家开源均未系统覆盖的三个方向，作为 SeptMuse 的核心差异化。

### 6.1 因果链记忆（Causal Memory Graph）

**问题**：现有图记忆只存事实三元组 `(subject)-[relation]->(object)`，缺因果。无法回答"如果当时没做 X 会怎样"。

**设计**：在图上新增因果边类型。

```python
class CausalEdge(SQLModel, table=True):
    id: str
    cause_event_id: str          # 指向 EpisodicEvent
    effect_event_id: str         # 指向 EpisodicEvent
    relation: str                # "enables" | "prevents" | "causes" | "inhibits"
    confidence: float
    counterfactual_valid: bool   # 是否已验证反事实
```

**抽取**：在 cognify 流水线增加 `extract_causal_edges` 阶段，LLM 从事件序列中识别因果。

**查询**：新增 `SearchType.CAUSAL_COMPLETION`，支持反事实查询"若 X 未发生，Y 是否仍发生"，走图遍历 + LLM 推理。

**验证计划**：构造已知因果的测试集，验证召回率与反事实推理准确率。

### 6.2 Ebbinghaus 遗忘曲线（Forgetting Curve + 主动复述）

**问题**：所有长时记忆永久等权，低价值记忆挤占检索；高价值记忆不被强化会衰减。

**设计**：每条长时记忆带强度字段，按遗忘曲线衰减；agent idle 时主动复述高价值低强度记忆。

```python
class MemoryStrength:
    memory_id: str
    strength: float = 1.0        # 当前强度 [0,1]
    last_accessed: datetime
    access_count: int = 0
    base_value: float            # 内禀价值 (规则/事实/偏好)

    def decay(self, now: datetime) -> float:
        # Ebbinghaus: R = exp(-t / S)
        # strength 随时间衰减, 访问时回升
        elapsed = (now - self.last_accessed).total_seconds()
        return math.exp(-elapsed / (self.base_value * S_FACTOR))

    def rehearse(self) -> None:
        # 主动复述: 访问一次, strength 回升 + access_count+1
        self.strength = min(1.0, self.strength + REHEARSAL_GAIN)
        self.access_count += 1
        self.last_accessed = now
```

**检索排序**：`final_score = relevance × strength`，低强度记忆自然下沉。

**主动复述**：agent idle/`Dream` 阶段，扫描 `strength < 0.3 且 base_value > 0.7` 的记忆，触发 rehearse（写入一个复述事件）。

**退化**：`strength < 0.1` 且持续 N 天无访问 → 归档到冷存储，不参与默认检索。

### 6.3 完整元认知自描述（Metacognitive Self-Description）

**问题**：ReMe meta 仅路由命名空间，agent 不"知道"自己记住了什么、记不住什么，无法自调策略。

**设计**：扩展元认知为三层，状态以语义记忆 + `meta` 标签持久化。

```python
class MetacognitionLayer:
    # L0: 命名空间索引 (借鉴 ReMe meta)
    #     列出所有可用记忆类型/目标, 驱动检索路由
    def read_meta(self) -> list[NamespaceDesc]: ...

    # L1: 记忆覆盖自描述 (自研, 真正的元认知)
    #     agent 定期生成"我记住了什么/记不住什么"的自述
    def generate_coverage_report(self) -> CoverageReport:
        # 扫描所有命名空间, 按主题/时间/置信度统计覆盖
        # 输出: "我清楚记得 alice 的编程偏好, 但对她的非技术兴趣覆盖薄弱"
        ...

    # L2: 策略自调 (自研)
    #     基于覆盖报告, 自调检索策略 (加深 / 换源 / 触发澄清提问)
    def adapt_strategy(self, report: CoverageReport) -> Strategy: ...
```

**持久化**：CoverageReport 存为语义记忆，打 `meta` + `coverage` 标签，跨会话累积。

**驱动**：检索前先查 L0 路由 + L1 覆盖，若覆盖薄弱 → L2 触发"主动澄清提问"或"加深检索"。

---

## 7. 借鉴 / 自研划分表

### 7.1 借鉴（直接复用或适配开源）

| 模块 | 借鉴源 | 复用方式 | 风险 |
|------|--------|---------|------|
| 工作记忆 Block | Letta `schemas/memory.py` `schemas/block.py` | 移植 schema + core_memory_* 工具 | 低, schema 简单 |
| 激活记忆 KVCache | MemOS `KVCacheMemory` | 仅自托管模型启用 | 中, 依赖 HF backend |
| 情节时序事件 | Zep/Graphiti Episode | 移植 Episode + reference_time 语义 | 中, Graphiti 独立部署较重 |
| 推理经验 | LangMem `Episode(obs/act/result)` | 直接采用 schema | 低 |
| 情节 raw log | Cass Episodic | 采用"统一 episodic 池"理念 | 低 |
| 语义抽取流水线 | Cognee cognify | 移植 classify→chunk→extract_graph→summarize | 中, LLM 调用成本 |
| 语义图三元组 | Graphiti `POST /fact` | 采用三元组 schema | 低 |
| 语义多信号检索 | mem0 | 采用四信号融合排序 | 低 |
| 语义多租户 | LangMem namespace 模板 | 直接采用 `("memories",org,user)` | 低 |
| 程序规则退化 | Cass Playbook helpful/harmful | 采用计数+deprecation schema | 低 |
| 文件记忆 | Basic Memory | 采用 Markdown+SQLite 双向同步 | 中, 同步冲突处理 |
| 捕获 hook | Agent Memory PostToolUse | 移植 hook 流水线 | 中, 需 agent 框架适配 |
| 检索渐进式 | ReMe 三层 | 采用 meta→向量→历史 渐进 | 低 |
| 检索混合 | Agent Memory BM25+向量+图 | 采用三路融合 | 低 |
| token 预算 | Agent Memory / Hermes | 采用注入前裁剪 | 低 |
| 写审批 | Hermes write_approval | 采用拦截器 | 低 |
| 隐私脱敏 | Agent Memory 隐私过滤 | 采用 hook 内脱敏 | 低 |
| 演化-链接生长 | A-MEM Zettelkasten | 采用 add 时自动找关系 | 中, ChromaDB 依赖 |
| 演化-反思升华 | Cass reflect+curate | 采用 Diary→Playbook 流程 | 低 |
| 演化-Dream | ReMe Dream | 采用空闲期批量 wikilink | 低 |
| 跨 agent 共享 | Agno user_id 共享 db | 采用同 db+user_id 模式 | 低 |
| 元认知路由 L0 | ReMe meta Layer 0 | 采用 meta 命名空间索引 | 低 |

### 7.2 自研（创新空白 + 编排层）

| 模块 | 自研理由 | 复杂度 | 验证方式 | 无 LLM 降级 |
|------|---------|:-----:|---------|-------------|
| **因果链记忆** | 14 家均无因果边 + 反事实查询 | 高 | 因果测试集召回率 + 反事实准确率 | 关键词规则抽取（"导致"/"因为"）+ 用户显式标注；反事实查询降级为纯图遍历返回路径，无 LLM 推理 |
| **Ebbinghaus 遗忘曲线** | 14 家均无强度衰减 + 主动复述 | 中 | 衰减曲线拟合 + 复述后强度回升 | 纯数学 exp(-t/S)，不需要 LLM；base_value 规则驱动（规则=0.8/事实=0.5/偏好=0.6） |
| **元认知 L1 覆盖自描述** | ReMe 仅 L0 路由, 无覆盖自述 | 中 | 覆盖报告准确性 + 策略自调效果 | 纯统计报告（"语义库 N 条,程序库 M 条"），无自然语言自述；离线生成 + 增量更新 |
| **元认知 L2 策略自调** | 全新, 基于覆盖报告驱动 | 中 | 澄清提问触发合理性 | 规则驱动（覆盖度 < 阈值 → 触发澄清提问；高频强覆盖 → 加深检索） |
| **源同步器** | 多形态共存一致性, 14 家均无统一方案 | 中 | 形态间漂移检测 + 补偿 | hash 比对漂移检测，不需要 LLM |
| **统一编排 API** | 六层记忆的统一入口 + 路由 | 中 | 端到端记忆回路集成测试 | V2 remember/recall/improve/forget 在无 LLM 时退化为 verbatim 存储 + 向量检索 + 规则演化 |
| **置信度 + 溯源** | Cass 仅规则退化, 事实/推断未区分 | 低 | 置信度加权排序 A/B | provenance 字段用户/规则标注，confidence 规则计算，不需要 LLM |
| **跨 agent 权限治理** | Agno 仅共享无权限, 需 RBAC | 中 | 权限矩阵测试 | RBAC 角色矩阵纯数据结构，不需要 LLM |

---

## 8. 技术选型（对齐 solarseptem-ai 生态）

| 层 | 选型 | 理由 |
|----|------|------|
| 后端框架 | FastAPI | solarseptem 标准 |
| ORM/数据模型 | SQLModel | solarseptem 标准 |
| 主数据库 | PostgreSQL | 复用平台后端 |
| 向量库 | pgvector (Postgres 扩展) | 减少依赖, 与主库同实例 |
| 图库 | Apache AGE (Postgres 图扩展) | 优先; 重场景才用 Neo4j |
| 文件索引 | SQLite (per-workspace) | Basic Memory 模式 |
| 全文检索 | BM25 (Postgres tsvector 或 SQLite FTS5) | 混合检索一路 |
| 自托管模型后端 | HuggingFace transformers (KVCache) | 仅激活/参数化记忆需要 |
| 前端 | Next.js | solarseptem 标准 |
| MCP 集成 | MCP server 暴露记忆工具 | 对齐 mcp_market 子系统 |

---

## 9. 演进路线（分阶段）

### 阶段 1：最小闭环（MVP）
**范围**：L1 工作 Block + L3b 语义记忆（向量+图）+ 被动捕获 + 混合检索
**目标**：agent 能记住用户偏好并在下一会话召回
**借鉴**：Letta Block + mem0 双存储 + Cognee cognify
**验证**：偏好记忆跨会话召回率 ≥ 80%

### 阶段 2：认知分层完整
**范围**：叠加 L3a 情节（Zep Episode + LangMem Episode）+ L3c 程序（Cass Playbook）+ 文件记忆（Basic Memory）
**目标**：四类内容类型齐备，人可读审计
**验证**：情节时序查询 + 程序规则退化流程跑通

### 阶段 3：横切关注点完整
**范围**：hook 捕获 + 渐进式检索 + 治理（token 预算/审批/脱敏）+ 演化（A-MEM/Cass/ReMe）+ 跨 agent 共享
**目标**：生产可用
**验证**：多 agent 共享记忆 + 编码 agent 零侵入捕获

### 阶段 4：创新增量
**范围**：因果链 + 遗忘曲线 + 元认知自描述 + 源同步器
**目标**：差异化能力落地
**验证**：三项各独立的测试集 + 端到端集成

### 阶段 5：激活/参数化（可选）
**范围**：MemOS KVCache + LoRA
**目标**：自托管高吞吐场景优化
**前提**：已接入自托管模型后端

---

## 10. 风险与权衡

| 风险 | 影响 | 缓解 |
|------|------|------|
| Graphiti 独立部署较重 | 阶段 2 依赖 | 优先用 Apache AGE 替代 |
| 多形态共存一致性 | 源同步器复杂 | 以图为权威源, 文件/向量补偿同步 |
| 因果边抽取准确率 | 创新价值 | LLM 抽取 + 人工校验种子集 |
| 遗忘曲线参数调优 | 强度失真 | 先用经典 Ebbinghaus 参数, A/B 调 S_FACTOR |
| 元认知 L1 报告成本 | 推理开销 | 离线生成, 增量更新而非全量 |
| 跨 agent 权限治理 | RBAC 复杂 | 阶段 3 先做读共享, 阶段 4 加写权限 |
| 激活/参数化仅限自托管 | 覆盖面 | 标记为可选, API 模型跳过 |

---

## 11. 模块边界与平台集成

### 11.1 与 solarseptem-ai 其它子系统的接口

| 子系统 | 接口 |
|--------|------|
| SolAgent | 主要消费方, 通过 SeptMuse API 读写记忆 |
| model_gateway | 提供 LLM 抽取/推理能力 |
| mcp_market | SeptMuse 暴露为 MCP server, 供 agent 调用记忆工具 |
| agent_runner | 编排 hook 捕获（PostToolUse 接入点） |
| SeptKit / SeptLex | 可选消费方（按需共享语义记忆） |

### 11.2 SeptMuse 对外 API 草案

```
POST   /memories/working/blocks/{label}      # 写工作记忆 block
GET    /memories/working/blocks              # 读所有 block (编译 XML)
PUT    /memories/working/blocks/{label}      # replace
POST   /memories/episodic                     # 写情节事件
POST   /memories/semantic                     # 写语义事实
POST   /memories/procedural                   # 写程序规则
POST   /memories/search                       # 统一检索 (元认知路由)
POST   /memories/search/causal                # 反事实因果查询
GET    /memories/meta/coverage                # 元认知覆盖报告
POST   /memories/rehearse                    # 主动复述触发
GET    /agents/{user_id}/memories            # 跨 agent 共享读
```

---

## 12. 验收清单

- [ ] 三维正交骨架在代码中表现为独立模块（content_types / storage_forms / concerns 三目录）
- [ ] 每个组合格子有明确借鉴源标注
- [ ] 每个组合格子标注 LLM 依赖度（必需/可降级/不需要），见 §2.2
- [ ] 每个自研项有"无 LLM 降级"方案，见 §7.2
- [ ] §1.2 含"零 LLM 可用"设计原则（第 6 条）
- [ ] 零 LLM 模式端到端可用：无 `SEPTMUSE_LLM` 时，remember/recall/improve/forget 不崩溃且降级路径可观测
- [ ] 三个创新空白各有独立测试集
- [ ] 因果链有"已知因果测试集"验证召回率 + 反事实准确率（§6.1）
- [ ] 遗忘曲线有 A/B 调 S_FACTOR 机制（§6.2）
- [ ] 元认知 L1 报告持久化为 SemanticFact(tags=["meta","coverage"])，跨会话累积（§6.3）
- [ ] 借鉴的开源库版本固定（opensource/ 下已克隆 mem0/letta/ReMe）
- [ ] 阶段 1 MVP 可独立验证跨会话偏好召回
- [ ] 文档自洽：无混轴、无重复类型、无形态当类型

---

## 附录 A：14 家调研结论摘要

| 库 | 唯一性贡献 | SeptMuse 借鉴点 |
|----|-----------|----------------|
| mem0 | 双存储托管黑盒 | 双存储 + 多信号检索 |
| Letta | Block 自编辑 + XML 注入 | 工作记忆 Block |
| ReMe | 文件原生 + Dream + meta L0 | 文件记忆 + 渐进检索 + 元认知 L0 |
| Cognee | LLM ETL cognify + GRAPH_COMPLETION | 语义抽取流水线 |
| Zep/Graphiti | 时序 Episode + 增量 + edge_type_map | 情节时序事件 |
| LangMem | Episode(obs/act/result) + namespace | 推理经验 + 多租户 |
| A-MEM | Zettelkasten 自演化 | 演化-链接生长 |
| MemOS | 全形态（含 KV-cache + LoRA） | 激活/参数化记忆 |
| Basic Memory | 人机双向共写 Markdown | 文件记忆 + 双向同步 |
| Cass | ACE 三层 + helpful/harmful 退化 | 程序规则退化 + 统一 episodic |
| Agent Memory | PostToolUse hook + 脱敏 + token 预算 | 捕获 hook + 治理 |
| Agno | user_id 跨 agent 共享 | 跨 agent 共享 |
| Hermes | 硬 token 上限 + 审批 + 多平台 | 治理（审批/上限） |
| Graphiti | ontology 声明 + 图命名空间 | 图存储独立内核 |

## 附录 B：三项真空白（14 家均未实现）

1. **因果链记忆** — SeptMuse §6.1 自研
2. **Ebbinghaus 遗忘曲线** — SeptMuse §6.2 自研
3. **完整元认知自描述**（L1 覆盖 + L2 策略自调）— SeptMuse §6.3 自研（L0 借鉴 ReMe）
