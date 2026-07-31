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
"""LLMService — LLM 服务 (借鉴 Langflow StoreService 模式)。

包装 LLM ABC, 提供:
- LLM provider 生命周期管理 + 延迟初始化
- 统一 complete 接口
- 可用性检查 + 降级策略
- 统计信息 (请求数 / 错误数)
"""

from __future__ import annotations

from septmuse.services.llm.factory import LLMServiceFactory
from septmuse.services.llm.service import LLMService

__all__ = [
    "LLMService",
    "LLMServiceFactory",
]
