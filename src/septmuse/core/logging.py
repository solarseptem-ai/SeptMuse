#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""SeptMuse 结构化日志模块

提供结构化日志功能（基于平台 solarseptem_core 同款实现，暂内建待迁移）：
- 带 ANSI 颜色的终端输出
- 滚动文件输出 — 普通日志 (app.log) 和错误日志 (error.log) 分离
- 基于文件大小的滚动，可配置最大文件大小
- 基于时间的旧日志文件清理
- 格式：[2026-05-29 16:50:11] [INFO] [timer.py:293] | Importing graph profiling

环境变量（平台统一前缀，迁移到 solarseptem_core 无成本）：
- ``SOLARSEPTEM_LOG_LEVEL``: 覆盖日志级别
- ``SOLARSEPTEM_LOG_DIR``: 覆盖日志目录

用法：
    from septmuse.core.logging import configure, get_logger

    configure(log_level="INFO", log_dir=Path("./logs"))
    logger = get_logger(__name__)
    logger.info("hello world")

注：本模块由平台 solarseptem_core.logging 迁移而来，故保留 SOLARSEPTEM_ 前缀；
待平台核心库建立后上移为 ``solarseptem_core.logging_utils``。
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
LEVEL_TO_INT = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_KEEP_DAYS = 30

# 终端彩色输出的 ANSI 转义码
_ANSI_COLORS: dict[str, str] = {
    "DEBUG": "\033[34m",  # 蓝色
    "INFO": "\033[32m",  # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",  # 红色
    "CRITICAL": "\033[1;31m",  # 深红（粗体红色）
}
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# 文件清理
# ---------------------------------------------------------------------------


def cleanup_old_logs(log_dir: Path, keep_days: int) -> int:
    """删除 *log_dir* 中超过 *keep_days* 天的日志文件。

    返回被删除的文件数量。
    """
    if not log_dir.exists():
        return 0
    cutoff = time.time() - keep_days * 86400
    deleted = 0
    for pattern in ("*.log", "*.log.*"):
        for f in log_dir.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
    return deleted


def _default_log_dir() -> Path:
    """计算合理的默认日志目录，不依赖第三方库。"""
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(base) / "solarseptem" / "logs"


# ---------------------------------------------------------------------------
# stdout 日志过滤器（分离普通 / 错误输出流）
# ---------------------------------------------------------------------------


class _MaxLevelFilter(logging.Filter):
    """只放行级别 **不高于** *max_level* 的日志记录。"""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class _MinLevelFilter(logging.Filter):
    """只放行级别 **不低于** *min_level* 的日志记录。"""

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


# ---------------------------------------------------------------------------
# 文件日志桥接
# ---------------------------------------------------------------------------

_SINKS_LOCK = Lock()
_ALL_FILE_SINKS: list[_FileLogSink] = []


class _FileLogSink:
    """持有两个 ``RotatingFileHandler`` 实例 —— 一个用于普通级别日志
    （DEBUG / INFO / WARNING，写入 ``app.log``），另一个用于错误级别日志
    （ERROR / CRITICAL，写入 ``error.log``）。

    构造时还会执行一次基于 *keep_days* 的清理。
    """

    def __init__(
        self,
        log_dir: Path,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        keep_days: int = DEFAULT_KEEP_DAYS,
    ) -> None:
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # 启动时清理过期日志
        cleanup_old_logs(log_dir, keep_days)

        self._handlers: list[logging.Handler] = []

        h_normal = self._make_handler(
            log_dir / "app.log",
            max_bytes,
            backup_count,
            [_MaxLevelFilter(logging.WARNING)],
        )
        h_error = self._make_handler(
            log_dir / "error.log",
            max_bytes,
            backup_count,
            [_MinLevelFilter(logging.ERROR)],
        )
        self._normal_handler = h_normal
        self._error_handler = h_error

    @staticmethod
    def _make_handler(
        path: Path,
        max_bytes: int,
        backup_count: int,
        filters: list[logging.Filter],
    ) -> logging.Handler:
        handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        for f in filters:
            handler.addFilter(f)
        return handler

    def __call__(self, _logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        formatted = event_dict.get("_formatted")
        if formatted is None:
            return event_dict
        level_name = event_dict.get("level", "info").upper()
        level_no = LEVEL_TO_INT.get(level_name, logging.INFO)
        if level_no >= logging.ERROR:
            self._error_handler.emit(
                logging.LogRecord(
                    name="septmuse",
                    level=level_no,
                    pathname="",
                    lineno=0,
                    msg=formatted,
                    args=(),
                    exc_info=None,
                )
            )
        else:
            self._normal_handler.emit(
                logging.LogRecord(
                    name="septmuse",
                    level=level_no,
                    pathname="",
                    lineno=0,
                    msg=formatted,
                    args=(),
                    exc_info=None,
                )
            )
        return event_dict

    def close(self) -> None:
        for h in (self._normal_handler, self._error_handler):
            with contextlib.suppress(Exception):
                h.close()


def _register_sink(sink: _FileLogSink) -> None:
    with _SINKS_LOCK:
        # 关闭并丢弃之前注册的 sink（例如上一次 configure() 调用留下的），避免文件句柄泄漏
        for old in _ALL_FILE_SINKS:
            old.close()
        _ALL_FILE_SINKS.clear()
        _ALL_FILE_SINKS.append(sink)


def _shutdown_sinks() -> None:
    """关闭所有已注册的文件处理器（可安全地多次调用）。"""
    with _SINKS_LOCK:
        for sink in _ALL_FILE_SINKS:
            sink.close()
        _ALL_FILE_SINKS.clear()


# 在进程正常退出时注册清理回调
atexit.register(_shutdown_sinks)


# ---------------------------------------------------------------------------
# 自定义 PrintLogger: 日志走 stderr (Unix 惯例 + 兼容 pytest capsys)
# ---------------------------------------------------------------------------
# structlog.PrintLogger 在 file != stdout 时直接持有 file 对象引用,
# 在 pytest capsys 下会拿到被关闭的原始 stderr 而报 "I/O operation on closed file";
# 这里改为每次 msg 调用时 print(file=sys.stderr) 解析当前 sys.stderr,
# 既符合日志走 stderr 的 Unix 惯例, 又支持 capsys 捕获分离 stdout/stderr,
# 同时保证 CLI 的 stdout (JSON/print) 不被日志污染, mcp stdio 协议通道也安全。


class _StderrPrintLogger:
    """structlog PrintLogger 兼容替代: 日志走当前 sys.stderr。

    每次调用解析 ``sys.stderr`` (而非持有固定引用), 支持 pytest capsys 替换。
    """

    def __init__(self) -> None:
        self._lock = Lock()

    def msg(self, message: str) -> None:
        with self._lock:
            print(message, file=sys.stderr, flush=True)

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class _StderrPrintLoggerFactory:
    """产 _StderrPrintLogger 的 factory (供 structlog.configure logger_factory)。"""

    def __call__(self, *args: Any) -> _StderrPrintLogger:
        return _StderrPrintLogger()


# ---------------------------------------------------------------------------
# structlog 处理器
# ---------------------------------------------------------------------------


def _format_record(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """构建规范的日志行。

    结果存储在 ``_formatted`` 中，下游处理器（文件 sink、终端渲染器）可以直接使用，无需重新格式化。
    """
    ts = event_dict.get("timestamp", "")
    level = event_dict.get("level", "info").upper()
    filename = event_dict.get("filename", "???")
    line = event_dict.get("lineno", 0)
    msg = event_dict.get("event", "")
    event_dict["_formatted"] = f"{ts} [{level} ] [{filename}:{line} ] | {msg}"
    return event_dict


def _render_console(_logger: Any, _method: str, event_dict: dict[str, Any]) -> str:
    """格式化终端输出，带 ANSI 彩色。

    直接返回字符串（当处理器链以字符串返回值结束时，structlog 的 PrintLogger 会按此处理）。
    """
    line = event_dict.get("_formatted", event_dict.get("event", ""))
    level = event_dict.get("level", "info").upper()
    color = _ANSI_COLORS.get(level, "")
    if color and sys.stderr.isatty():
        return f"{color}{line}{_RESET}"
    return line


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

# 单一配置标志 (合并原 _log_configured / _configured_once, 幂等守卫)
_configured = False


def configure(
    log_level: str = "INFO",
    log_dir: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    keep_days: int = DEFAULT_KEEP_DAYS,
    *,
    file_output: bool = True,
    console_output: bool = True,
    force: bool = False,
) -> None:
    """配置 SeptMuse 的结构化日志。

    参数
    ----------
    log_level:
        最低日志级别。可选值：``DEBUG``、``INFO``、``WARNING``、``ERROR``、
        ``CRITICAL``。可以被环境变量 ``SOLARSEPTEM_LOG_LEVEL`` 覆盖。
    log_dir:
        日志文件目录。为 ``None`` 时，依次回退到
        ``SOLARSEPTEM_LOG_DIR`` 环境变量，然后是平台默认日志目录。
        设置 ``file_output=False`` 可完全禁用文件日志。
    max_bytes:
        单个日志文件滚动前的最大大小（字节）。
    backup_count:
        保留的滚动文件数量（``app.log.1`` … ``app.log.N``）。
    keep_days:
        超过此天数的滚动日志文件在启动时被删除。
    file_output:
        设为 ``False`` 则不输出文件日志（仅终端输出）。
    console_output:
        设为 ``False`` 则不向 stdout / stderr 写入（仅文件输出）。
    force:
        设为 ``True`` 强制重新配置（即使已配置过）。默认 ``False`` 时，
        首次调用生效，后续调用幂等跳过——避免重复注册 sink 导致文件句柄泄漏。
    """

    global _configured
    if _configured and not force:
        return
    _configured = True

    # 解析日志级别（环境变量覆盖参数）
    env_level = os.getenv("SOLARSEPTEM_LOG_LEVEL", "").upper()
    if env_level in VALID_LOG_LEVELS:
        log_level = env_level
    level_no = LEVEL_TO_INT.get(log_level.upper(), logging.INFO)

    # 解析日志目录
    if file_output:
        if log_dir is not None:
            log_dir = Path(log_dir)
        elif env_dir := os.getenv("SOLARSEPTEM_LOG_DIR", ""):
            log_dir = Path(env_dir)
        else:
            log_dir = _default_log_dir()
    else:
        log_dir = None

    # ---- 构建处理器链 ----

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="[%Y-%m-%d %H:%M:%S ]"),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        _format_record,
    ]

    # 文件 sink
    if file_output and log_dir is not None:
        sink = _FileLogSink(log_dir, max_bytes, backup_count, keep_days)
        _register_sink(sink)
        shared_processors.append(sink)  # type: ignore[arg-type]

    # 终端渲染 —— 始终以返回字符串的处理器结尾
    if console_output:
        shared_processors.append(_render_console)
    else:

        def _quiet(_l: Any, _m: str, ed: dict[str, Any]) -> str:
            return ed.get("_formatted", ed.get("event", ""))

        shared_processors.append(_quiet)

    # ---- 组装所有组件 ----

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        context_class=dict,
        logger_factory=_StderrPrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """返回指定 *name* 的 structlog BoundLogger（通常传 ``__name__``）。"""
    return structlog.get_logger(name)


def shutdown() -> None:
    """关闭所有文件日志处理器。

    可用于测试和优雅关闭钩子。已通过 ``atexit`` 自动注册，
    因此在生产环境中手动调用是可选的。
    """
    _shutdown_sinks()


def is_configured() -> bool:
    """返回日志系统是否已配置（用于测试断言）。"""
    return _configured


__all__ = [
    "cleanup_old_logs",
    "configure",
    "get_logger",
    "is_configured",
    "shutdown",
]


# ---------------------------------------------------------------------------
# 引导：使用合理的默认值配置，让 `get_logger` 无需显式调用 configure() 即可工作。
# 用户代码之后仍可调用 configure(force=True) 来覆盖默认设置。
# ---------------------------------------------------------------------------

# 首次导入时以默认值配置一次 (幂等: _configured 守卫保证不重复)
configure(
    log_level=os.getenv("SOLARSEPTEM_LOG_LEVEL", "DEBUG"),
    file_output=False,
    console_output=True,
)
logger: structlog.BoundLogger = structlog.get_logger()
