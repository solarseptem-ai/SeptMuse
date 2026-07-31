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
"""SeptMuse — agent 记忆系统。

三维正交架构: 内容类型 × 存储形态 × 横切关注点。
零配置: ``pip install septmuse`` 即可用, 默认 SQLite 组合后端 + HashEmbedder (无需 API key)。

快速开始:
    from septmuse import Memory

    m = Memory()
    m.add("我喜欢 Python 和 vim 键位", user_id="alice")
    results = m.search("alice 喜欢什么编辑器", user_id="alice", top_k=3)
    for r in results:
        print(r["memory"], r["score"])

三入口 (CLI / REST / MCP):
    - CLI:   ``septmuse init && septmuse add "..." --user alice && septmuse search "..." --user alice``
    - REST:  ``septmuse serve`` → http://localhost:8000/docs
    - MCP:   ``septmuse mcp`` (stdio) 或挂载到 FastAPI SSE/HTTP transport

更多示例: examples/quickstart.py
"""

from septmuse.configs.defaults import MemoryConfig
from septmuse.core.logging import configure, get_logger, shutdown
from septmuse.memory import Memory

__version__ = "0.1.0"

__all__ = ["Memory", "MemoryConfig", "__version__", "configure", "get_logger", "shutdown"]
