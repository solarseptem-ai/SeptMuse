# Task 4 报告: Memory 新增 remember/recall/forget/improve + V2Memory 降级薄层

## 状态: DONE

## 一句话总结
Memory 类已含 4 编排方法 (remember/recall/forget/improve 委托 self.add/search/delete), V2Memory 已降级为 DeprecationWarning 薄层委托, 消除双 facade 分裂。

## 文件变更
| 文件 | 变更 |
|------|------|
| `src/septmuse/memory/main.py` | Memory.__init__ 初始化 V2 编排组件 (working_memory/token_budget/meta/evolution/forgetting/capture); 新增 remember/recall/forget/improve 4 方法 (L710-944); 新增 _create_working_memory_store/_persist_coverage/_load_coverage_report/_profile_to_prompt 辅助 |
| `src/septmuse/memory/memory_v2.py` | V2Memory 重写为 DeprecationWarning 薄层委托, 透传 mem 属性 + remember/recall/forget/improve 委托 self.mem; 保留 _normalize_messages |
| `tests/unit/test_memory_orchestration.py` | 10 测试覆盖 remember 委托 add、空文本、去重、recall id 保留、token 预算、forget 委托 delete、improve 运行、V2 弃用警告、V2 委托 remember/recall |

## 测试结果
- `tests/unit/test_memory_orchestration.py`: **10 passed** (2.72s)
- `tests/unit/test_v2_memory.py` (回归): **18 passed** (2.92s, 17 个 DeprecationWarning 符合预期)

## Ruff
- `ruff check src/septmuse/memory/main.py src/septmuse/memory/memory_v2.py`: **All checks passed!**

## 实现要点
1. **委托模式**: remember→self.add, recall→self.search, forget→self.delete, 复用完整通路 (实体抽取+缓存失效+决策+reranker), 不平行重组
2. **编排独有叠加**: remember 叠加 episodic raw_log + working_memory block; recall 叠加遗忘曲线加权 + token 预算(保留 id) + L0 路由 + L2 策略自调 + block/规则注入; forget 叠加双时态 invalidate + 图边清理
3. **容错**: 编排组件 try/except, 失败降级为 None 不阻塞基础 add/search/delete
4. **V2Memory 兼容**: 透传 mem 属性 (store/embedder/typed_store/semantic/episodic/procedural/capture/token_budget/meta/evolution/forgetting/retrieval/causal), 旧测试直接访问 v2.* 不破

## 顾虑
- 无。QdrantClient.__del__ 在 Python 关闭时的 ImportError 是已知析构噪声 (qdrant-client 本地嵌入模式), 不影响测试结果。
