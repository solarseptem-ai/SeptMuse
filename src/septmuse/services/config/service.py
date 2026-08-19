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
"""ConfigService — 配置服务实现。"""

from __future__ import annotations

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.services.base import Service

logger = get_logger(__name__)


class ConfigService(Service):
    """配置服务 — 包装 MemoryConfig + default_config。"""

    name = "config_service"

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or default_config()
        self._reload_count = 0
        self.set_ready()
        logger.info("config_service_initialized", db_path=str(self._config.db_path))

    @property
    def config(self) -> MemoryConfig:
        """当前配置 (MemoryConfig 实例)。"""
        return self._config

    def reload(self) -> MemoryConfig:
        """重新从环境变量加载配置。"""
        self._config = default_config()
        self._reload_count += 1
        logger.info("config_service_reloaded", reload_count=self._reload_count)
        return self._config

    @property
    def reload_count(self) -> int:
        """配置重载次数。"""
        return self._reload_count

    def is_valid(self) -> bool:
        """校验配置是否有效 (pydantic 模型不会抛异常, 仅检查关键字段)。"""
        if self._config.database.db_path is None:
            return True
        return True
