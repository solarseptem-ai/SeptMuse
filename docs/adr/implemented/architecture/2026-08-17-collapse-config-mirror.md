# Agent Note: 塌缩 config mirror —— 删除 per-backend 配置类层

Status: implemented

## Problem

`configs/` 原有两层结构：7 个 `Base*Config` 基类 + 每个 backend 一个 config 子类（如 `HashEmbedderConfig`、`OnnxEmbedderConfig` 等，共 54 文件）+ `enums.py`（94 行）+ `database.py`。

问题：

1. **per-backend config 类从未被实例化** —— `registry.BackendEntry.config_cls` 字段指向它们，但 `providers.resolve()` 实际走 `model_dump(exclude_none=True)`，`config_cls` 分支只在 `config_cls is None and not kwargs` 时短路，从未真正构造 config 实例。
2. **config 已 drift** —— `HashEmbedderConfig.dims=384` vs `HashEmbedder.dim=128`；impl 的 `DEFAULT_*` 常量是唯一活源，config 类是死镜像。
3. **819 行死代码** —— 54 文件 + `enums.py` + `database.py` + 7 个子目录，全部无人调用。

## Decision

**纯删除 + 合并**：

- 删除 54 个 per-backend config `*.py` + `configs/enums.py` + `configs/database.py` + 7 个子目录
- 合并 7 个 `Base*Config` + `DatabaseConfig` 进 `configs/base.py`（单文件 ~380 行）
- 删除 `registry.BackendEntry.config_cls` 字段
- 删除 `providers._config_to_kwargs()` 的 `config_cls` 分支
- impl 的 `DEFAULT_*` 常量成为唯一配置真源

## Alternatives considered

- **wiring `config_cls`**：让 registry 真正构造 config 实例。否决：config 类已 drift，wiring 会把死值变活值，反而更危险。
- **移 defaults 到 registry**：把 `DEFAULT_*` 常量移到 registry 条目。否决：registry 变配置文件，违反"impl 是唯一真源"。
- **enum 替换**：用 enum 替代 `enums.py`。否决：enums 本身就是死代码，删除 > 替换。
- **factory dicts**：用 dict 替代 config 类。否决：增加一个新层，不减少死代码。
- **flatten `MemoryConfig`**：把所有配置压进 `MemoryConfig`。否决：`MemoryConfig` 已经是顶层入口，问题在 per-backend 层。
- **纯删除 + 合并**：选中。最小 diff，消除死代码，不增加新层。

## Consequences

**正面**：

- 消除 819 行死代码 + config drift
- 单一真源：impl `DEFAULT_*` 常量
- registry 条目从 4 字段降到 3 字段（`module`/`cls`/`deps`）

**负面**：

- per-backend 配置定制能力降低 —— 但 YAGNI，从未被使用，需要时按当时需求新建（不从删除代码恢复）

**中性**：

- `Base*Config` 类保留在 `base.py`，作为 pydantic schema 仍有用（如果未来需要从环境变量批量构造配置）
