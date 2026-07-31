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
"""MCP 请求上下文 — contextvars 传递 user_id / client_name。

源码参考 mem0/openmemory/api/app/mcp_server.py 的 user_id_var / client_name_var 模式。

http/sse 模式: 由 URL 路径参数 set (transports.py 处理)
stdio 模式: 由环境变量 SEPTMUSE_USER_ID 初始化 (server.run_stdio)
工具: 优先用显式 user_id 参数, 缺省回退 contextvar (兼容两种模式)
"""

from __future__ import annotations

import contextvars

# 用户 ID (跨 agent 共享键, 必填)
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="")

# client 名称 (调用方标识, 如 "claude-code" / "cursor")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name", default="")

# agent ID (可选, 区分同 user 多 agent)
agent_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_id", default="")
