# Learnings — 失败归档与模式识别

记录 AI 幻觉、Bug、返工（同一问题修改两次以上或方案被拒绝后重做），标注失败类型和规避方法。与 CHANGELOG 分工：CHANGELOG 记"改了什么"，本文档记"为什么出错、如何规避"。

---

## 2026-07-23 · 测试 mock 与去重特性冲突 (Task 2: MMRReranker)

**失败类型**：边界遗漏（测试 mock 设计缺陷）

**现象**：`test_top_k_truncation` 期望 `top_k=2` 返回 2 条结果，实际返回 1 条。`test_dedup_high_similarity` 则正确验证了去重特性（2 条相同 embedding → 1 条）。

**根因**：`test_top_k_truncation` 的 `MockEmbedder` 对所有输入返回相同向量 `[1.0, 0.0, 0.0]`，导致 5 条结果两两余弦相似度 = 1.0 > 0.9，触发 MMRReranker 的去冗余逻辑（"相似度 >0.9 的结果只保留排名靠前的一个"）。该测试名为"top_k 截断"，实际却测了去重，与 `test_dedup_high_similarity` 的预期矛盾——同一套相同 embedding 无法既"去重到 1"又"截断到 2"。

**规避方法**：
- 测试单一特性时，mock 必须隔离该特性。测 `top_k` 截断应使用**正交/互异** embedding 避免触发去重；测去重应使用**相同** embedding。
- 提供测试 mock 时，先推演 mock 数据在待测算法下会产生什么副作用（去重、过滤、归一化等），避免一个 mock 同时触发多个特性导致断言矛盾。
- 当两个测试用相同 mock 却期望不同结果时，必然存在特性冲突，需拆分 mock 隔离场景。

---

## 2026-07-23 · 任务规格对 Entity dataclass 字段描述错误 (Task 5/6: Entity Boost)

**失败类型**：AI 幻觉（规格描述与实际符号不符）

**现象**：`test_entity_boost_increases_score` 等测试失败，`entity_boost_failed` 警告被触发，entity_boost 始终为 0.0。任务规格声称 `Entity` dataclass 字段为 `text, entity_type, metadata, embedding`，但实际为 `text, entity_type, start, end`。测试 mock 构造 `Entity(text="Python", entity_type="TOPIC")` 缺少必填的 `start`/`end`，在 `entity_extractor.extract()` 内部抛 TypeError，被 HybridRetriever 的 try/except 吞掉并记为 `entity_boost_failed`。

**根因**：任务规格对内部符号 `Entity` 的字段描述与代码实际不符（描述了 `metadata`/`embedding`，实际是 `start`/`end`）。未先用 CodeGraph 核对 dataclass 真实签名，直接信任规格描述编写 mock。

**规避方法**：
- 编写涉及项目内 dataclass/class 构造的 mock 前，**必须先用 CodeGraph 查询该符号的真实字段签名**，禁止凭规格描述假设字段名。
- 当 try/except 吞错并记 warning 时（如 `entity_boost_failed`），优先怀疑内部构造失败而非业务逻辑缺陷；用 `python -c` 单步复现 mock 构造可快速定位。
- 规格描述的"借鉴来源字段"（如 mem0 的 Entity 有 metadata/embedding）≠ SeptMuse 的实际字段，两者命名可能因架构调整而不同。
