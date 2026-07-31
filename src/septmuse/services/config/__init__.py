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
"""ConfigService — 配置服务 (借鉴 Langflow SettingsService 模式)。

包装 MemoryConfig + default_config, 提供:
- 统一配置加载 / 校验 / 重载
- 环境变量解析入口
- 供其他服务 (EmbedderService / LLMService) 依赖
"""

from __future__ import annotations

from septmuse.services.config.factory import ConfigServiceFactory
from septmuse.services.config.service import ConfigService

__all__ = [
    "ConfigService",
    "ConfigServiceFactory",
]
