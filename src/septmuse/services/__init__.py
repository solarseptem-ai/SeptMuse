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
"""SeptMuse 服务层。

目录结构:
    base.py        Service ABC (name + ready + get_schema + teardown)
    schema.py      ServiceType 枚举 (SeptMuse 7 类核心服务)
    factory.py     ServiceFactory 基类 (绑定 service_class + create)
    manager.py     ServiceManager (注册 + 懒创建 + 单例缓存)
    deps.py        依赖注入入口 (每个服务一个 get_xxx_service() 函数)

对外入口:
    from septmuse.services import get_service, get_service_manager, ServiceType
    from septmuse.services.deps import get_memory_service  # 后续实现

模式:
    deps.py get_xxx_service()
      → get_service(ServiceType.XXX, XxxFactory())
        → ServiceManager.get()
          → 首次: Factory.create() + 缓存
          → 后续: 直接返回缓存单例
"""

from __future__ import annotations

from septmuse.services.base import Service
from septmuse.services.factory import ServiceFactory
from septmuse.services.manager import ServiceManager, get_service_manager
from septmuse.services.schema import ServiceType

__all__ = [
    "Service",
    "ServiceFactory",
    "ServiceManager",
    "ServiceType",
    "get_service_manager",
]
