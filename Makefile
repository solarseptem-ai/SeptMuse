# SeptMuse Makefile — 开发、测试、Docker 部署一站式命令
# 参考: mem0 / graphiti / MemOS 的 Makefile 模式
#
# 用法:
#   make install        # 安装开发依赖
#   make check          # format + lint + test (提交前必跑)
#   make docker-up      # Docker 一键启动 (SQLite 零配置)
#   make docker-up-pg   # Docker 启动 (PostgreSQL + pgvector)

.PHONY: help install install-dev install-prod clean format lint typecheck \
        test test-unit test-e2e test-cov test-single build check \
        serve mcp cli-init \
        docker-build docker-up docker-up-pg docker-up-mysql \
        docker-down docker-down-pg docker-down-mysql \
        docker-down-clean docker-down-clean-pg docker-down-clean-mysql \
        docker-logs docker-logs-pg docker-logs-mysql

# ── 变量 ──────────────────────────────────────────────────────────────────
PYTHON      := python
PYTHONPATH  := src
PYTEST      := PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest
RUFF        := $(PYTHON) -m ruff
PIP         := $(PYTHON) -m pip
DOCKER_COMPOSE := docker compose -f docker/docker-compose.yml
DOCKER_COMPOSE_PG := $(DOCKER_COMPOSE) -f docker/docker-compose.pg.yml
DOCKER_COMPOSE_MYSQL := $(DOCKER_COMPOSE) -f docker/docker-compose.mysql.yml

# ── 默认目标 ──────────────────────────────────────────────────────────────
help:
	@echo "SeptMuse Makefile"
	@echo ""
	@echo "  开发:"
	@echo "    make install        安装开发依赖 (dev + server)"
	@echo "    make format         格式化代码 (ruff)"
	@echo "    make lint           Lint 检查 (ruff)"
	@echo "    make typecheck      类型检查 (mypy)"
	@echo "    make test           运行全部测试 (unit + e2e)"
	@echo "    make test-unit      仅单元测试"
	@echo "    make test-e2e       仅端到端测试"
	@echo "    make test-cov       测试 + 覆盖率报告"
	@echo "    make test-single T=path::func  单测"
	@echo "    make check          format + lint + test (提交前必跑)"
	@echo "    make build          构建 wheel 包"
	@echo "    make clean          清理缓存和构建产物"
	@echo ""
	@echo "  运行:"
	@echo "    make serve          启动 REST API (localhost:8000)"
	@echo "    make mcp            启动 MCP Server (stdio)"
	@echo "    make cli-init       初始化本地数据库"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build       构建镜像"
	@echo "    make docker-up          SQLite 零配置启动"
	@echo "    make docker-up-pg       PostgreSQL + pgvector 启动"
	@echo "    make docker-up-mysql    MySQL 8 启动"
	@echo "    make docker-down        停止 (SQLite)"
	@echo "    make docker-down-pg     停止 (PostgreSQL)"
	@echo "    make docker-down-mysql  停止 (MySQL)"
	@echo "    make docker-down-clean  停止 + 删除数据卷 (SQLite)"
	@echo "    make docker-down-clean-pg     停止 + 删除数据卷 (PG)"
	@echo "    make docker-down-clean-mysql  停止 + 删除数据卷 (MySQL)"
	@echo "    make docker-logs        查看日志 (SQLite)"
	@echo ""

# ── 安装 ──────────────────────────────────────────────────────────────────
install: install-dev

install-dev:
	$(PIP) install -e ".[dev,server]"

install-prod:
	$(PIP) install ".[server,postgres,mysql]"

# ── 清理 ──────────────────────────────────────────────────────────────────
clean:
	rm -rf dist build *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf tmp
	rm -f .coverage .coverage.*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── 格式化 ────────────────────────────────────────────────────────────────
format:
	$(RUFF) check --select I --fix src/ tests/ examples/
	$(RUFF) format src/ tests/ examples/

# ── Lint ──────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/ examples/
	$(RUFF) format --check src/ tests/ examples/

# ── 类型检查 ──────────────────────────────────────────────────────────────
typecheck:
	$(PYTHON) -m mypy src/septmuse

# ── 测试 ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/unit/ tests/e2e/ -q

test-unit:
	$(PYTEST) tests/unit/ -q

test-e2e:
	$(PYTEST) tests/e2e/ -q

test-cov:
	$(PYTEST) tests/unit/ tests/e2e/ -q \
		--cov=src/septmuse \
		--cov-report=term-missing \
		--cov-report=html:cov-report

test-single:
	$(PYTEST) $(T) -q

# ── 检查 (提交前必跑) ─────────────────────────────────────────────────────
check: format lint test
	@echo "check passed"

# ── 构建 ──────────────────────────────────────────────────────────────────
build:
	$(PYTHON) -m build

# ── 运行 ──────────────────────────────────────────────────────────────────
serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m septmuse.cli.main serve --host 0.0.0.0 --port 8000 --with-rest

mcp:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m septmuse.api.mcp.server

cli-init:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m septmuse.cli.main init

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	docker build -t septmuse:latest .

docker-up:
	$(DOCKER_COMPOSE) up -d --build

docker-up-pg:
	$(DOCKER_COMPOSE_PG) up -d --build

docker-up-mysql:
	$(DOCKER_COMPOSE_MYSQL) up -d --build

docker-down:
	$(DOCKER_COMPOSE) down

docker-down-pg:
	$(DOCKER_COMPOSE_PG) down

docker-down-mysql:
	$(DOCKER_COMPOSE_MYSQL) down

docker-down-clean:
	$(DOCKER_COMPOSE) down -v

docker-down-clean-pg:
	$(DOCKER_COMPOSE_PG) down -v

docker-down-clean-mysql:
	$(DOCKER_COMPOSE_MYSQL) down -v

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-logs-pg:
	$(DOCKER_COMPOSE_PG) logs -f

docker-logs-mysql:
	$(DOCKER_COMPOSE_MYSQL) logs -f