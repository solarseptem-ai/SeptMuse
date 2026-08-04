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
"""DatabaseServiceFactory — 数据库服务工厂。

用法:
    factory = DatabaseServiceFactory()
    svc = factory.create(config)
    engine = svc.get_engine()
"""

from __future__ import annotations

from septmuse.configs.base import MemoryConfig
from septmuse.services.database.service import DatabaseService
from septmuse.services.factory import ServiceFactory


class DatabaseServiceFactory(ServiceFactory):
    """数据库服务工厂 — 支持注入自定义 MemoryConfig 或使用 default_config。"""

    def __init__(self) -> None:
        super().__init__(DatabaseService)

    def create(self, config: MemoryConfig | None = None) -> DatabaseService:
        return DatabaseService(config=config)
