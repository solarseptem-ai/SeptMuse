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
"""服务工厂基类 (借鉴 Langflow services/factory.py ServiceFactory)。

每个具体 Service 的 Factory 继承此类, 设 service_class + 覆盖 create()。

用法:
    class MemoryServiceFactory(ServiceFactory):
        def __init__(self):
            super().__init__(MemoryService)

        def create(self, config=None) -> MemoryService:
            return MemoryService(config or default_config())

    factory = MemoryServiceFactory()
    service = factory.create()
"""

from __future__ import annotations

from typing import Any

from septmuse.services.base import Service


class ServiceFactory:
    """服务工厂基类——绑定 service_class, create() 创建实例。

    Langflow 模式: Factory 负责实例化 Service, ServiceManager 负责缓存。
    deps.py 里每个 get_xxx_service() 传 default factory 给 get_service()。
    """

    def __init__(self, service_class: type[Service] | None = None) -> None:
        self.service_class = service_class

    def create(self, *args: Any, **kwargs: Any) -> Service:
        """创建 Service 实例 (子类可覆盖以注入依赖)。"""
        if self.service_class is None:
            raise ValueError("service_class not set on factory subclass")
        return self.service_class(*args, **kwargs)
