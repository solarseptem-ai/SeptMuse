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
"""EmbedderService — 嵌入服务。

包装 Embedder ABC, 提供:
- 模型生命周期管理 + 延迟初始化
- 统一 embed / embed_batch 接口
- 统计信息 (请求次数 / 后端 / 维度)
"""

from __future__ import annotations

from septmuse.services.embedder.factory import EmbedderServiceFactory
from septmuse.services.embedder.service import EmbedderService

__all__ = [
    "EmbedderService",
    "EmbedderServiceFactory",
]
