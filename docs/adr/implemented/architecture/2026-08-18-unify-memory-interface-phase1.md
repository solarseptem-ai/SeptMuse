# Agent Note: 统一 Memory 接口 Phase 1 — 删 4 死壳入口

Status: implemented

## Problem

`septmuse.memory` 包导出 6 个入口类（Memory/V2Memory/MemOS/MemCube/MemoryRegistry/AsyncMemory），但生产代码（api/operations.py、api/mcp/server.py、cli/main.py）只走 `Memory`。其余 5 个中 4 个零生产调用（V2Memory/MemOS/MemCube/MemoryRegistry），全是测试引用。多个入口造成理解困难——用户不知道该用哪个，每个入口的 interface 几乎一样但行为微妙不同。

## Decision

**Phase 1**（本期）：删除 4 个零生产调用的入口类 + 适配测试。

1. **删 `memory_v2.py` (V2Memory)** — 已标 DEPRECATED，委托 Memory，76 行壳。2 个测试文件适配为直接测 Memory。
2. **删 `os.py` (MemOS)** — 三层壳顶（MemOS→MemCube→Memory），83 行，4 方法纯转发。test_mem_os.py 删除（测壳无价值）。
3. **删 `cube.py` (MemCube)** — type 路由壳，被 MemOS 包，91 行。
4. **删 `registry.py` (MemoryRegistry)** — 存了 `self.registry` 但从不调任何方法，dispatch 是硬编码 if/elif，79 行死代码。
5. **`__init__.py` 导出从 11 符号降到 5** — 只导出 Memory + AsyncMemory + ABC（LongTermMemory/MemoryABC/ShortTermMemory）。
6. **AsyncMemory 保留** — 有独立 async 语义 + 3 个测试文件。Phase 4 再并入 Memory。

## Alternatives considered

- **全删 5 个（含 AsyncMemory）** — 拒绝：AsyncMemory 有独立 async 语义和 3 个测试文件，分阶段降低风险。
- **保留 MemCube 的 type 路由** — 拒绝：MemCube 的 type 路由（semantic/episodic/procedural）在 Memory facade 已有 `add_typed/add_fact/add_episode/add_rule` 方法覆盖。
- **适配 MemOS/MemCube 测试为测 Memory** — 拒绝：测壳的转发行为无价值，直接删。
- **V2Memory 测试适配为测 Memory** — 接受：V2Memory 测的是 remember/recall/improve/forget 编排，这些方法在 Memory 上存在，适配后测试逻辑不变。

## Consequences

- `from septmuse.memory import V2Memory/MemOS/MemCube/MemoryRegistry` 不再可用（breaking change）。
- `from septmuse.memory import Memory` 仍是主入口。
- `test_mem_os.py` 删除（16 tests）。`test_v2_memory.py` 适配为测 Memory（15 tests 保留）。`test_memory_orchestration.py` 删 3 个 V2Memory 测试，保留 7 个 Memory 测试。`test_memory_abc.py` V2Memory→Memory（1 test 适配）。
- 本期是 4 期重构的第 1 期。后续：②移工厂到 registry_v2 → ③委托 60 方法到领域模块 → ④加 async 到 Memory + 删 AsyncMemory。
