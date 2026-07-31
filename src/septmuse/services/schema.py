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
"""服务类型枚举 (借鉴 Langflow services/schema.py ServiceType)。

SeptMuse 核心服务类型——每个对应一个 Service + ServiceFactory + deps.py get 函数。
后续实现时, 在 services/<name>/ 下建 factory.py + service.py, deps.py 暴露 get 函数。
"""

from __future__ import annotations

from enum import Enum


class ServiceType(str, Enum):
    """SeptMuse 服务类型枚举。"""

    CONFIG_SERVICE = "config_service"
    MEMORY_SERVICE = "memory_service"
    EMBEDDER_SERVICE = "embedder_service"
    LLM_SERVICE = "llm_service"
    RETRIEVAL_SERVICE = "retrieval_service"
    INGESTION_SERVICE = "ingestion_service"
    GOVERNANCE_SERVICE = "governance_service"
