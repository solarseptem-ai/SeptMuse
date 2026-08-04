#!/bin/bash
# SeptMuse Docker Entrypoint — 等待数据库就绪后启动服务
# 支持: SQLite (跳过等待) / PostgreSQL / MySQL
set -e

DB_URL="${SEPTMUSE_DB_URL:-}"

# ── 等待数据库就绪 ────────────────────────────────────────────────────────

wait_for_postgres() {
    local host="${1:-postgres}"
    local port="${2:-5432}"
    echo "[entrypoint] 等待 PostgreSQL ${host}:${port} ..."
    local max_retries=30
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('${host}', ${port}))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            echo "[entrypoint] PostgreSQL 就绪"
            return 0
        fi
        retry=$((retry + 1))
        echo "[entrypoint] PostgreSQL 未就绪, 重试 ${retry}/${max_retries} ..."
        sleep 2
    done
    echo "[entrypoint] 错误: PostgreSQL 等待超时"
    exit 1
}

wait_for_mysql() {
    local host="${1:-mysql}"
    local port="${2:-3306}"
    echo "[entrypoint] 等待 MySQL ${host}:${port} ..."
    local max_retries=30
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('${host}', ${port}))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            echo "[entrypoint] MySQL 就绪"
            return 0
        fi
        retry=$((retry + 1))
        echo "[entrypoint] MySQL 未就绪, 重试 ${retry}/${max_retries} ..."
        sleep 2
    done
    echo "[entrypoint] 错误: MySQL 等待超时"
    exit 1
}

# ── 解析 DB_URL 并等待 ────────────────────────────────────────────────────

if [ -n "$DB_URL" ]; then
    # 提取协议和主机
    if echo "$DB_URL" | grep -qi "^postgresql://"; then
        host=$(echo "$DB_URL" | sed -n 's|^postgresql://[^@]*@\([^:/]*\).*|\1|p')
        port=$(echo "$DB_URL" | sed -n 's|^postgresql://[^@]*@[^:]*:\([0-9]*\).*|\1|p')
        wait_for_postgres "${host:-postgres}" "${port:-5432}"
    elif echo "$DB_URL" | grep -qi "^mysql://"; then
        host=$(echo "$DB_URL" | sed -n 's|^mysql://[^@]*@\([^:/]*\).*|\1|p')
        port=$(echo "$DB_URL" | sed -n 's|^mysql://[^@]*@[^:]*:\([0-9]*\).*|\1|p')
        wait_for_mysql "${host:-mysql}" "${port:-3306}"
    elif echo "$DB_URL" | grep -qi "^mysql+pymysql://"; then
        host=$(echo "$DB_URL" | sed -n 's|^mysql+pymysql://[^@]*@\([^:/]*\).*|\1|p')
        port=$(echo "$DB_URL" | sed -n 's|^mysql+pymysql://[^@]*@[^:]*:\([0-9]*\).*|\1|p')
        wait_for_mysql "${host:-mysql}" "${port:-3306}"
    else
        echo "[entrypoint] DB_URL 已设但协议未知, 跳过等待: ${DB_URL%%://*}://***"
    fi
else
    echo "[entrypoint] SQLite 模式, 跳过 DB 等待"
fi

# ── 启动服务 ──────────────────────────────────────────────────────────────

echo "[entrypoint] 启动 SeptMuse: $*"
exec "$@"