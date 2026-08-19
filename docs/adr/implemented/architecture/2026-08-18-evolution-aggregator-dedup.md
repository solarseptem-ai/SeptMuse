# Agent Note: 演化聚合器去重 — facade 4 方法委托 EvolutionModule + scheduler 委托

Status: implemented

## Problem

`Memory` facade 的 `reflect/dream/resolve_conflicts/deduplicate_entities` 4 个方法内联构造 `SessionReflector/DreamIntegrator/ConflictResolver`，和 `EvolutionModule` 的同名方法构造完全相同。`EvolutionScheduler` 的 `conflict/distillation/graph` 3 个任务也内联构造相同聚合器。三处写同样的编排逻辑，改一处要同步三处。此外 `EvolutionModule.compress` 构造新 `Summarizer` 每次调用，facade 已有 `self.summarizer` 组件复用，EvolutionModule.compress 零生产调用方。

## Decision

1. **facade 4 方法全委托** (Q1=A) — `reflect/dream/resolve_conflicts/deduplicate_entities` 改为 `return self.evolution.X(...)`，和 `improve()` 委托一致。
2. **统一 EvolutionModule 返回格式** (Q2=A) — facade `reflect` 返回 `{"proposed", "accepted", "rule_ids"}` 改为 EvolutionModule 的 `{"lessons_proposed", "lessons_accepted", "rule_ids"}`。facade 委托后直接透传，零映射代码。2 个测试适配 key 名。
3. **统一 EvolutionModule 优雅守卫** (Q3=A) — facade `dream` 的 `assert graph_store` 改为 EvolutionModule 的 `if graph_store is None: return zeros`。委托后自动获得。
4. **scheduler 3 方法委托** (Q4=A) — `conflict/distillation/graph` 3 个任务改为 `self.memory.evolution.X(...)`，`ForgettingRetriever.archive_stale` 保留内联（遗忘管理不是演化聚合）。
5. **删 EvolutionModule.compress** (Q5=C) — facade 已有 `self.summarizer` 组件复用，EvolutionModule.compress 零生产调用方。

## Alternatives considered

- **只委托 resolve_conflicts + deduplicate_entities** (Q2=B) — 拒绝：返回格式差异在 Q2 统一解决后无理由保留内联。
- **保留 assert** (Q3=B) — 拒绝：scheduler 后台任务里 assert 会炸，优雅守卫是正确行为。
- **scheduler 保留内联** (Q4=B) — 拒绝：3 个构造和 EvolutionModule 完全相同，不委托就是纯重复。
- **EvolutionModule.compress 保留** (Q5=B) — 拒绝：零调用方 + facade 已有 summarizer 组件。
- **facade 做 key 映射** (Q2=C) — 拒绝：多一层映射代码，委托后直接透传更简洁。

## Consequences

- facade `reflect` 返回 key 从 `proposed/accepted` 变为 `lessons_proposed/lessons_accepted`（breaking change for callers 硬编码 key）。
- facade `dream` 不再 assert graph_store，graph_store=None 时返回 `{"processed": 0, "links_created": 0}`。
- `EvolutionModule.compress` 删除，facade `Memory.compress` 保留（委托 `self.summarizer`）。
- scheduler 3 个后台任务走 facade 委托链路，多一层间接调用（可接受——后台任务非热路径）。
- `test_facade_integration.py:102` 和 `test_distill.py:201-202` 适配 key 名。
