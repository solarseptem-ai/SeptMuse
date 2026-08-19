# Agent Note: CognifyPipeline seam 修复 — 关系边 CRUD 并入 EntityStore + m007 迁移追踪

Status: implemented

## Problem

CognifyPipeline 直接访问 EntityStore 的私有 `_engine` 属性获取裸 SQLAlchemy 连接，绕过 ORM 层用裸 SQL 建 `entity_relations` 表并执行 CRUD。三处问题：

1. **seam 泄漏** — `entity_store._engine.raw_connection()` 访问私有属性，EntityStore 的 interface 不应暴露 `_engine`
2. **迁移系统外建表** — `entity_relations` 和 `septmuse_entities` 两张表不在 m001-m006 迁移里，`schema_version` 不追踪
3. **重复建表** — `EntityRelationTable` ORM model 已存在于 `services/database/models/entity.py:39-53`，`EntityStore.from_engine()` 调 `SQLModel.metadata.create_all(engine)` 已建表；CognifyPipeline `_init_relations_table()` 用不同 DDL 第二次建同一张表

## Decision

1. **关系边 CRUD 并入 EntityStore**（Q1=A）— EntityStore 已拥有 `septmuse_entities`（顶点），加上关系边（边）成为完整实体图谱仓储。新增 `store_relation(source, relation, target, *, user_id, memory_id)` 和 `get_neighbors(entity, *, user_id)` 两个方法，用 `Session(engine)` + `EntityRelationTable` ORM 操作。
2. **保留 EntityStore.from_engine 的 create_all**（Q2=A，从 B 回退）— `create_all` 幂等：ORMMemoryStore 已建表时为 no-op，独立使用时建表。移除后 test_entity_store_orm + test_entity_batch 的独立 `from_engine` 用法全崩。m007 迁移仍保留用于 `schema_version` 追踪。
3. **m007 迁移覆盖两张表**（Q6=B）— 加 `septmuse_entities` + `entity_relations` 到迁移系统，`check_table` 跳过已有表只记版本。
4. **CognifyPipeline 无状态化**（Q3=A, Q8=A）— 删 `self._conn` / `self._lock` / `self._init_relations_table()`，构造函数参数不变。`_store_relation` 改调 `entity_store.store_relation()`，`get_entity_neighbors` 改调 `entity_store.get_neighbors()`。
5. **读路径不经管线**（Q4=A, Q9=A）— `Memory.get_entity_relations()` 直接调 `entity_store.get_neighbors()`，不再构造 CognifyPipeline。
6. **EntityStore 新方法用原始字段参数**（Q7=A）— 不依赖 extraction 层的 `Triplet` 类型，CognifyPipeline 负责拆字段。
7. **EntityModule 不动**（Q10=A）— 仍调 Memory facade，facade 内部改实现。

## Alternatives considered

- **新建 RelationStore**（Q1=B）— 独立仓储管关系边。拒绝：顶点和边是一个图谱的两个半，分开管破坏 locality，又多一个 store 类。
- **留在 CognifyPipeline 但换 ORM**（Q1=C）— 只把裸 SQL 换成 Session。拒绝：CognifyPipeline 是"管线"模块却持久化关系边，职责混乱。
- **只覆盖 entity_relations 迁移**（Q6=A）— 拒绝：两张表都加成本很低，完整追踪更好。
- **CognifyPipeline 单例化**（Q3=B）— 拒绝：复用实例是 R3-J（god constructor）的范围，不在这次改。
- **EntityModule 直接调 EntityStore**（Q10=B）— 拒绝：facade 的价值就是隐藏内部 store 切换。

## Consequences

- EntityStore interface 加 2 个方法（`store_relation`, `get_neighbors`），从纯顶点仓储变为图谱仓储。
- CognifyPipeline 不再持有数据库连接，回到纯管线编排角色。
- `entity_relations` 和 `septmuse_entities` 在 `schema_version` 表有版本记录，旧 DB 升级可追踪。
- 3 个生产构造点中 main.py:935 读路径构造点删除，剩余 2 个写路径保留。
- `test_cognify.py` 6 个直接测试保留，验证 relation 存储时改查 EntityStore 而非 CognifyPipeline._conn。
