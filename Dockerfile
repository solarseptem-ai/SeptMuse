# SeptMuse Docker 镜像 — 零配置 SQLite + HashEmbedder, 离线可用
# 用法:
#   docker build -t septmuse:latest .
#   docker run -p 8000:8000 -e SEPTMUSE_API_KEY=sk-xxx septmuse:latest
#   # 或 docker compose -f docker/docker-compose.yml up (见 docker/docker-compose.yml)

FROM python:3.11-slim

# 元数据
LABEL org.opencontainers.image.title="SeptMuse" \
      org.opencontainers.image.description="Agent 记忆系统 — 三维正交架构, 零配置开箱即用" \
      org.opencontainers.image.source="https://github.com/sonhhxg0529/solarseptem-ai" \
      org.opencontainers.image.licenses="Apache-2.0"

# 工作目录
WORKDIR /app

# 先复制依赖描述 + 源码, 再安装 (利用 Docker layer cache)
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY examples/ ./examples/

# 安装: 核心 + server + 数据库驱动 (postgres + mysql)
# HashEmbedder 默认, 无需 sentence-transformers 模型下载
# 同时安装 pg/mysql 驱动, 运行时通过 SEPTMUSE_DB_URL 切换后端
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[server,postgres,mysql]"

# ── 非 root 用户 (安全最佳实践) ──────────────────────────────────────────
RUN groupadd -r septmuse -g 1000 \
    && useradd -r -g septmuse -u 1000 -m -d /home/septmuse septmuse \
    && mkdir -p /data /home/septmuse/.septmuse \
    && chown -R septmuse:septmuse /app /data /home/septmuse

# 数据卷: SQLite 数据库持久化
VOLUME ["/data"]

# 环境变量默认值
ENV SEPTMUSE_DB_PATH=/data/septmuse.db \
    SEPTMUSE_EMBEDDER=hash \
    SEPTMUSE_INFER=false \
    PYTHONUNBUFFERED=1

# 启动入口: 等待 DB 就绪后启动服务
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# 暴露 REST + MCP 端口
EXPOSE 8000

# 切换到非 root 用户
USER septmuse

# 启动: 等待 DB → 启动服务
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["septmuse", "serve", "--host", "0.0.0.0", "--port", "8000", "--with-rest"]