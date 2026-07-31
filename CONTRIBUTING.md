# Contributing to SeptMuse

感谢你对 SeptMuse 感兴趣！本文档说明如何参与贡献。

## 开发环境

### 依赖

- **Python**：3.10+（CI 测试 3.10 / 3.11 / 3.12）
- **构建后端**：hatchling
- **Linter / Formatter**：ruff（line-length 120）
- **Test framework**：pytest + pytest-asyncio

### 初始化

```bash
git clone <repo-url>
cd SeptMuse
pip install -e ".[all]"
pre-commit install  # 可选但推荐
```

`[all]` extra 包含所有可选 provider + server + dev 依赖。

## 开发流程

### 1. 分支

```bash
git checkout -b feat/<slug>      # 新功能
git checkout -b fix/<slug>       # bug 修复
git checkout -b docs/<slug>      # 文档
```

### 2. 改动

- 遵循现有代码风格（ruff line-length 120，snake_case 文件名，test_<module>.py 测试文件）。
- 新增 provider 必须放可选 extra，**禁止**加入核心 `dependencies`。
- 新增依赖需用 try/except ImportError 守护，并 raise 清晰的"install extras X"提示。

### 3. 测试

```bash
# 全回归（跳过 FastAPI 版本冲突的 rbac 测试）
$env:PYTHONPATH="src"
python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes"

# 仅 e2e
python -m pytest tests/ -m e2e -q

# 单文件
python -m pytest tests/unit/test_memory.py -v
```

**测试保护规则**（来自 AGENTS.md）：

- 现有全部单元测试、接口测试案例固定不动，严禁修改测试代码、调整断言、删减用例来规避业务缺陷。
- 所有测试不通过的问题，仅通过优化业务逻辑、补齐容错处理、完善参数校验、修复分支漏洞解决。
- 仅可新增测试用例覆盖新功能，原有存量测试一律保留原始逻辑。
- 代码交付前校验：不存在注释测试、跳过测试、篡改测试参数等绕过行为。

### 4. Lint + Format

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### 5. 提交

Conventional Commits：

```
feat: 新增 LoRA 参数化记忆
fix: HashEmbedder 中文分词错误
docs: 更新 README
refactor: 抽取 register_routes
test: 补充 e2e 跨会话召回测试
chore: 升级依赖
```

### 6. PR Checklist

- [ ] 代码通过 `ruff check` 和 `ruff format`
- [ ] 全量测试通过（576 passed + 9 skipped）
- [ ] 新增功能有对应测试覆盖
- [ ] 公开 API 变更需更新文档（`docs/specs/`）
- [ ] 新依赖加入可选 extra，**不**加入核心 `dependencies`
- [ ] 提交信息遵循 Conventional Commits
- [ ] 不修改现有测试用例来规避缺陷

## 架构约束

SeptMuse 遵循**三维正交架构**（见 `docs/specs/agent-memory-architecture.md`）：

```
平面A 内容类型: 工作 / 情节 / 语义 / 程序 / 身份
平面B 存储形态: block / 向量 / 图 / 文件 / 激活 / 参数化
平面C 横切关注点: 捕获 / 检索 / 治理 / 演化 / 共享 / 元认知
```

新增功能必须能放入这三个平面的某一格，否则需先在 architecture 文档中讨论扩展。

## 添加新 Provider

参考已有 provider（如 `src/septmuse/providers/embedders/openai.py`）：

1. 在对应目录创建 `src/septmuse/<category>/<provider>.py`，继承 `base.py`。
2. 配置加入 `src/septmuse/configs/defaults.py`。
3. 在 `src/septmuse/<category>/__init__.py` 注册。
4. 添加 `tests/unit/test_<provider>.py` 测试（外部 HTTP / 模型加载必须 mock）。
5. 新依赖加入 `pyproject.toml` 的对应 optional group。

## 不做

- 不修改 `pyproject.toml` 核心 `dependencies`（仅可选 extra）。
- 不提交 `.env`、API key、模型权重。
- 不跳过 pre-commit hook。
- 不 force-push 到 `main`。
- 不修改 CI/CD workflow 除非显式批准。
- 不修改现有测试断言来规避缺陷（见"测试保护规则"）。

## License

贡献的代码在 [Apache 2.0 License](./LICENSE) 下发布。提交即表示同意该协议。
