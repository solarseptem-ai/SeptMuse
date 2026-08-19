# Agent Note: 工程纪律体系调整 —— 文档分层 + Agent Notes 四态 + 项目级 Skills

Status: implemented

## Problem

SeptMuse 的 AGENTS.md 是"百科全书"风格（~250 行混 standing orders + 架构细节 + 测试怪癖 + 环境变量表 + 框架怪癖），没有文档分层，没有机器可检查的 invariant。决策记录只有 ADR 二态（Accepted），无法捕获"为什么没做 X"。项目级知识靠全局 skills，不知道 SeptMuse 的项目特定约定。

对比 deepseek-harness 的工程纪律体系：文档分层（tier taxonomy，one home per fact）+ Agent Notes 四态生命周期 + 项目级 Skills + 机器可检查 gate + slop checklist + prose standard。

## Decision

6 个决策：

1. **决策记录形式**：`docs/adr/{lifecycle}/{class}/` 四态子目录（`proposed`/`implemented`/`rejected`/`archived`），三类分类（`architecture`/`simplification`/`process`）。ADR-0001 迁入 `implemented/architecture/`。格式 gate P2 再做。

2. **术语表**：CONTEXT.md 保留在根目录 + 加 anchor + 交叉链接。不迁移到 docs/glossary.md。

3. **AGENTS.md 瘦身**：架构入口段→`docs/architecture.md`（Mermaid 图 + 数据流 + 设计决策表）；测试怪癖段→`docs/testing.md`；环境变量表保留压缩为表格；框架怪癖段保留压缩为 standing orders（1-2 行/rule + 链接到源码注释）。

4. **项目级 Skills**：创建 `.agents/skills/septmuse-code-review/`，编码 SeptMuse 项目特定约定（hash embedder 测试、MCP tools.py 禁 `from __future__`、SQLite `:memory:` 陷阱、ruff Windows bug）。先做 1 个验证模式。

5. **机器 gate**：P2 再做。当前靠纪律 + AGENTS.md + ruff check 够用。

6. **`/improve-codebase-architecture` 全局 skill 调整**：Step 1 加 slop checklist + corpus 区分（production vs non-production）+ supersession 检查（补充性，不改流程）。Step 2 暂不做。

## Alternatives considered

- **D1: `.agents/notes/` 全局迁移**：否决——破坏 `docs/adr/` 现有路径，SeptMuse 是单包项目不需要 monorepo 级 `.agents/notes/`。
- **D1: 扁平 + Status 字段**：否决——不可导航，无法按生命周期浏览。
- **D2: 迁移到 docs/glossary.md**：否决——CONTEXT.md 在根目录更可发现，domain-modeling 技能默认读 CONTEXT.md，迁移只换路径不增价值。
- **D3: 框架怪癖拆到 subsystems/**：否决——需要创建整个 subsystems/ 体系，P2 工作，不在本次范围。
- **D3: 框架怪癖拆到 Package README**：否决——同上，需要创建 README 体系。
- **D4: 一次做 3 个项目级 skill**：否决——先做 1 个验证模式，ponytail 不做 speculative skills。
- **D5: 现在做 gate**：否决——没有 scripts/ 目录，先验证前面的调整再做 gate。
- **D6: Step 2 流程性改动**：否决——先做 Step 1，验证效果后再决定 Step 2。
- **D6: 项目级 override**：否决——会导致两套行为不一致。

## Consequences

**正面**：

- AGENTS.md 从百科全书瘦身到 standing orders，架构/测试/reference 各有 home
- ADR 四态可以捕获 proposed/rejected，不只是 Accepted
- 项目级 skill 编码 SeptMuse 特定约定，PR 审查时有针对性
- 全局 skill 加 slop checklist 防止架构审查报告腐化

**负面**：

- AGENTS.md 瘦身是大迁移，需要仔细编辑（风险：遗漏 standing orders 或 link 断裂）
- 四态子目录没有格式 gate，靠人工遵守（P2 补 gate）
- 项目级 skill 只有 1 个，其他需求仍靠全局

**中性**：

- 框架怪癖保留压缩在 AGENTS.md，等 P2 拆到 subsystems/ 时再迁移
