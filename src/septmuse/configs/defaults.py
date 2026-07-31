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
"""SeptMuse 零配置默认 — BaseSettings 自动从 env + yaml + default 合并。

优先级: init kwargs > YAML > 代码默认; 旧版扁平 env (SEPTMUSE_*) 由
MemoryConfig._flat_env_aliases 兼容。零配置: SQLite + HashEmbedder (离线)。
"""

from __future__ import annotations

from septmuse.configs.base import MemoryConfig


def default_config() -> MemoryConfig:
    """零配置默认 — BaseSettings 自动从 env + yaml + default 合并。"""
    return MemoryConfig()
