# Honcho 数据流图

> 基于源码分析绘制，覆盖三大核心流程：记忆写入、记忆召回、后台记忆整合。

---

## 1. 记忆写入流（Write / Ingestion）

> 用户/Agent 存储消息 → 落库 → 入队 → 后台 Deriver 异步处理 → 生成结论

```mermaid
flowchart LR
    subgraph Client["客户端 / Agent"]
        C["SDK / HTTP Client"]
    end

    subgraph APIServer["Honcho API Server (FastAPI)"]
        R["Routers<br/>(sessions.py, messages.py)"]
    end

    subgraph Storage["存储层 (PostgreSQL + pgvector)"]
        MSG["messages 表"]
        EMB["message_embeddings 表<br/>(sync_state='pending')"]
        QUEUE["queue_items 表<br/>(任务队列)"]
    end

    subgraph Background["后台 Deriver Worker"]
        QM["QueueManager<br/>(轮询消费)"]
        DER["Deriver<br/>(minimal_deriver)"]
        LLM["LLM 调用<br/>(结构化输出)"]
        OUT["结论输出处理"]
    end

    subgraph VectorStore["向量存储 (pgvector)"]
        COLL["collections 表<br/>(observer, observed)"]
        DOC["documents 表<br/>(向量 embedding)&quot;]
        OBS["conclusions 表<br/>(explicit / deductive)&quot;]
    end

    C --&gt;|POST /sessions/{id}/messages| R
    R --&gt;|1. 落盘| MSG
    R --&gt;|2. 初始化 embedding| EMB
    R --&gt;|3. 入队 representation 任务| QUEUE

    QM --&gt;|4. 消费| DER
    DER --&gt;|5. 构建 prompt| LLM
    LLM --&gt;|6. 返回结论| OUT
    OUT --&gt;|7a. upsert| COLL
    OUT --&gt;|7b. upsert| DOC
    OUT --&gt;|7c. insert| OBS

    style Background fill:#e1f5e1,stroke:#333
    style VectorStore fill:#fff3e0,stroke:#333
```

### 写入流详细步骤

| 步骤 | 组件 | 动作 | 说明 |
|------|------|------|------|
| 1 | **Client** | `session.add_messages([...])` | SDK 批量发送消息（最多 100 条） |
| 2 | **Router** | 验证 + 落盘 | FastAPI 处理，写入 `messages` 表 |
| 3 | **Router** | 初始化 Embedding | `MessageEmbedding` 行写入，`sync_state='pending'` |
| 4 | **Router** | 入队 | `enqueue.py` 插入 `queue_items`，类型为 `representation` |
| 5 | **QueueManager** | 轮询消费 | Deriver Worker 独立进程，消费队列 |
| 6 | **Deriver** | 批处理 | 按 session 分组消息，构建 LLM prompt |
| 7 | **LLM** | 结构化抽取 | 输出 `explicit` 和 `deductive` 结论 |
| 8 | **Output Handler** | 持久化 | 结论按 `(observer, observed)` 写入 collections + documents + conclusions |

---

## 2. 记忆召回流（Read / Query / Dialectic）

> 用户查询 → Dialectic Agent 工具循环 → 多源检索 → LLM 生成回答

```mermaid
flowchart LR
    subgraph ClientQ[&quot;客户端 / Agent&quot;]
        CQ[&quot;peer.chat('Alice 喜欢什么？')&quot;]
    end

    subgraph APIServerQ[&quot;Honcho API Server&quot;]
        RQ[&quot;Router: peers.py /chat&quot;]
    end

    subgraph DialecticAgent[&quot;Dialectic Agent (同步工具循环)&quot;]
        DA[&quot;DialecticAgent<br/>(core.py)&quot;]
        TOOLS[&quot;工具集 (7-8 个)&quot;]
        LLMQ[&quot;LLM 调用&quot;]
    end

    subgraph SearchSources[&quot;检索源&quot;]
        MEM[&quot;search_memory<br/>(向量搜索结论)&quot;]
        MSGQ[&quot;search_messages<br/>(全文搜索消息)&quot;]
        OBSQ[&quot;get_observation_context<br/>(结论上下文)&quot;]
        GREP[&quot;grep_messages<br/>(关键词匹配)&quot;]
        TEMPORAL[&quot;search_messages_temporal<br/>(时序过滤)&quot;]
        CHAIN[&quot;get_reasoning_chain<br/>(推理链回溯)&quot;]
    end

    subgraph StorageQ[&quot;PostgreSQL / pgvector&quot;]
        COLL_Q[collections/documents]
        MSG_Q[messages]
        OBS_Q[conclusions + reasoning_trees]
    end

    subgraph FinalAnswer[&quot;最终响应&quot;]
        STREAM[&quot;SSE Stream<br/>或 同步 JSON&quot;]
    end

    CQ --&gt;|POST /peers/{id}/chat| RQ
    RQ --&gt;|初始化| DA
    DA --&gt;|循环调用| TOOLS
    TOOLS --&gt;|检索| MEM
    TOOLS --&gt;|检索| MSGQ
    TOOLS --&gt;|检索| OBSQ
    TOOLS --&gt;|检索| GREP
    TOOLS --&gt;|检索| TEMPORAL
    TOOLS --&gt;|检索| CHAIN

    MEM --&gt;|向量查询| COLL_Q
    MSGQ --&gt;|FTS / 向量| MSG_Q
    OBSQ --&gt;|JOIN| OBS_Q
    GREP --&gt;|ILIKE| MSG_Q
    TEMPORAL --&gt;|时间过滤| MSG_Q
    CHAIN --&gt;|树遍历| OBS_Q

    COLL_Q --&gt;|相关结论| TOOLS
    MSG_Q --&gt;|相关消息| TOOLS
    OBS_Q --&gt;|推理链| TOOLS

    TOOLS --&gt;|汇总上下文| LLMQ
    LLMQ --&gt;|生成回答| STREAM

    style DialecticAgent fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style SearchSources fill:#fff9c4,stroke:#f57f17
```

### Dialectic Agent 工具循环详解

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant DA as DialecticAgent
    participant LLM as LLM
    participant TS as Tool Search
    participant PG as PostgreSQL

    C-&gt;&gt;API: POST /peers/{id}/chat
    API-&gt;&gt;DA: agentic_chat(query, ...)
    DA-&gt;&gt;DA: 初始化系统提示

    loop Tool Loop (最多 N 轮)
        DA-&gt;&gt;LLM: 发送当前上下文
        LLM-&gt;&gt;DA: 返回工具调用意图
        DA-&gt;&gt;DA: 解析 tool_call

        alt 调用 search_memory
            DA-&gt;&gt;TS: 向量搜索结论
            TS-&gt;&gt;PG: SELECT ... ORDER BY embedding &lt;-&gt; query
            PG-&gt;&gt;TS: 返回相关结论
            TS-&gt;&gt;DA: 结论文本
        else 调用 search_messages
            DA-&gt;&gt;TS: 混合搜索消息
            TS-&gt;&gt;PG: vector + FTS hybrid
            PG-&gt;&gt;TS: 返回消息列表
            TS-&gt;&gt;DA: 消息文本
        else 调用 get_observation_context
            DA-&gt;&gt;TS: 获取结论完整上下文
            TS-&gt;&gt;PG: 查 conclusions + documents
            PG-&gt;&gt;TS: 完整上下文
            TS-&gt;&gt;DA: 上下文文本
        end

        DA-&gt;&gt;DA: 拼接 tool 结果到上下文
    end

    DA-&gt;&gt;LLM: 最终回答请求
    LLM-&gt;&gt;DA: 自然语言回答
    DA-&gt;&gt;API: 返回响应
    API-&gt;&gt;C: SSE Stream / JSON
```

### Dialectic 工具集

| 工具名 | 用途 | 检索目标 |
|--------|------|----------|
| `search_memory` | 向量语义搜索 | conclusions (documents 表) |
| `search_messages` | 混合搜索消息 | messages + message_embeddings |
| `get_observation_context` | 获取结论完整上下文 | conclusions + documents |
| `grep_messages` | 关键词模糊匹配 | messages (ILIKE) |
| `search_messages_temporal` | 时序范围搜索 | messages (created_at 范围) |
| `get_reasoning_chain` | 推理链回溯 | reasoning_trees |
| `get_recent_observations` | 获取最近结论 | conclusions (按时间排序) |

---

## 3. 后台记忆整合流（Dream / Consolidation）

> Dreamer 调度器 → 选择高价值结论 → Deduction/Induction Specialist → 更新推理树

```mermaid
flowchart TB
    subgraph Scheduler[&quot;Dreamer Scheduler&quot;]
        DSCHED[&quot;定期触发 / 手动触发&quot;]
    end

    subgraph Prioritization[&quot;优先级计算&quot;]
        SURP[&quot;Surprisal 计算<br/>(surprisal.py)&quot;]
        SELECT[&quot;选择 top-K 结论&quot;]
    end

    subgraph Deduction[&quot;Deduction Specialist (第一阶段)&quot;]
        DS1[&quot;输入: explicit conclusions&quot;]
        DS2[&quot;LLM 推理&quot;]
        DS3[&quot;产出: deductive conclusions<br/>+ 推理链链接&quot;]
    end

    subgraph Induction[&quot;Induction Specialist (第二阶段)&quot;]
        IS1[&quot;输入: explicit + deductive&quot;]
        IS2[&quot;LLM 归纳推理&quot;]
        IS3[&quot;产出: inductive conclusions<br/>+ Peer Card 更新&quot;]
    end

    subgraph Storage[&quot;PostgreSQL 更新&quot;]
        CONC[conclusions 表<br/>新增/更新/软删除]
        TREE[reasoning_trees 表<br/>premise -&gt; conclusion 链接]
        CARD[peer_cards 表<br/>Peer 身份快照]
    end

    DSCHED --&gt; SURP
    SURP --&gt; SELECT
    SELECT --&gt; DS1
    DS1 --&gt; DS2
    DS2 --&gt; DS3
    DS3 --&gt; IS1
    IS1 --&gt; IS2
    IS2 --&gt; IS3
    IS3 --&gt; CONC
    IS3 --&gt; TREE
    IS3 --&gt; CARD

    style Scheduler fill:#fce4ec,stroke:#c2185b
    style Prioritization fill:#f3e5f5,stroke:#7b1fa2
    style Deduction fill:#e8f5e9,stroke:#2e7d32
    style Induction fill:#e3f2fd,stroke:#1565c0
```

### Dreamer 处理流程

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **Surprisal 计算** | 所有结论 | 高 surprisal 的待处理结论 | 衡量信息的新颖度和重要性 |
| **Deduction** | explicit 结论 | deductive 结论 + 推理链 | 从具体事实推导隐含信息 |
| **Induction** | explicit + deductive | inductive 结论 + Peer Card | 归纳模式，更新身份快照 |
| **持久化** | 新生成结论 | conclusions / reasoning_trees / peer_cards | 软删除过时结论，建立推理树 |

---

## 4. 整体数据流全景图

```mermaid
flowchart TB
    subgraph Users[&quot;用户 / Agent&quot;]
        U1[&quot;写入消息&quot;]
        U2[&quot;查询问答&quot;]
    end

    subgraph Server[&quot;Honcho API Server (FastAPI)&quot;]
        S1[&quot;Sessions/Messages Router&quot;]
        S2[&quot;Peers Router (/chat)&quot;]
        S3[&quot;Conclusions Router&quot;]
    end

    subgraph DB[&quot;PostgreSQL + pgvector&quot;]
        T1[&quot;workspaces&quot;]
        T2[&quot;peers&quot;]
        T3[&quot;sessions&quot;]
        T4[&quot;messages&quot;]
        T5[&quot;message_embeddings&quot;]
        T6[&quot;collections&quot;]
        T7[&quot;documents (vector)&quot;]
        T8[&quot;conclusions&quot;]
        T9[&quot;reasoning_trees&quot;]
        T10[&quot;peer_cards&quot;]
        T11[&quot;queue_items&quot;]
    end

    subgraph Workers[&quot;后台 Worker 集群&quot;]
        W1[&quot;Deriver&quot;]
        W2[&quot;Summarizer&quot;]
        W3[&quot;Dreamer&quot;]
        W4[&quot;Reconciler&quot;]
    end

    U1 --&gt;|POST /messages| S1
    S1 --&gt; T4
    S1 --&gt; T5
    S1 --&gt; T11

    U2 --&gt;|POST /chat| S2
    S2 --&gt;|工具循环检索| T7
    S2 --&gt;|工具循环检索| T8
    S2 --&gt;|工具循环检索| T4
    S2 --&gt; U2

    T11 --&gt;|消费| W1
    W1 --&gt;|写入| T6
    W1 --&gt;|写入| T7
    W1 --&gt;|写入| T8

    T4 --&gt;|触发| W2
    W2 --&gt;|写入| T4

    T8 --&gt;|触发| W3
    W3 --&gt;|更新| T8
    W3 --&gt;|更新| T9
    W3 --&gt;|更新| T10

    T5 --&gt;|触发| W4
    W4 --&gt;|更新| T5

    style Workers fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style DB fill:#fff3e0,stroke:#ef6c00
```

---

## 5. 关键设计特点总结

| 设计点 | 说明 |
|--------|------|
| **读写分离** | 写入同步（落库），认知处理异步（Deriver/Dreamer） |
| **Queue 解耦** | API Server 只负责入队，Worker 负责重活，避免 HTTP 阻塞 |
| **多层级检索** | Dialectic Agent 可切换 7+ 种工具，按需检索不同数据源 |
| **推理链追踪** | `reasoning_trees` 记录结论的前提和推论，支持可解释性回溯 |
| **Surprisal 驱动** | Dreamer 优先处理高信息量的结论，而非简单按时间处理 |
| **Peer-Centric** | 所有结论按 `(observer, observed)` 存储，天然支持多视角建模 |

---

## 6. 与 Mem0 的数据流对比

| 维度 | **Honcho** | **Mem0** |
|------|-----------|----------|
| **写入流** | 消息 → 队列 → Deriver 异步抽取 | 记忆 → 同步抽取+存储 |
| **查询流** | Dialectic Agent（工具循环） | `search()` 直接返回记忆列表 |
| **后台处理** | Deriver + Summarizer + Dreamer + Reconciler | 无独立后台进程 |
| **检索深度** | 7+ 种工具，多源混合检索 | `search()` + 可选 reranker |
| **推理链** | 显式追踪（reasoning_trees） | 隐式（linked_memory_ids） |
| **架构复杂度** | 高（API + Worker 分离） | 低（库/服务一体） |

---

*图表绘制时间：2026-07-30*
*参考源码：opensource/honcho/src/*
