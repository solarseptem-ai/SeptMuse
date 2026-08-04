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
"""服务管理器。

注册工厂 → 懒创建 → 单例缓存。deps.py 通过 get_service_manager() 获取全局单例,
再 get(ServiceType.XXX) 取具体服务。

用法:
    manager = get_service_manager()
    manager.register_factory(ServiceType.MEMORY_SERVICE, MemoryServiceFactory())
    service = manager.get(ServiceType.MEMORY_SERVICE)
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.services.base import Service
from septmuse.services.factory import ServiceFactory
from septmuse.services.schema import ServiceType

logger = get_logger(__name__)


class ServiceManager:
    """服务管理器——注册工厂 + 懒创建 + 单例缓存。

    - register_factory: 注册 ServiceType → ServiceFactory 映射
    - get: 首次调用 factory.create() 并缓存, 后续直接返回缓存
    - are_factories_registered: 检查是否已注册 (deps.py 惰性初始化用)
    """

    def __init__(self) -> None:
        self._factories: dict[ServiceType, ServiceFactory] = {}
        self._services: dict[ServiceType, Service] = {}

    def register_factory(self, service_type: ServiceType, factory: ServiceFactory) -> None:
        """注册服务工厂。"""
        self._factories[service_type] = factory
        logger.debug("factory_registered", service_type=service_type.value)

    def register_factories(self, factories: dict[ServiceType, ServiceFactory]) -> None:
        """批量注册服务工厂。"""
        self._factories.update(factories)
        logger.debug("factories_registered", count=len(factories))

    def are_factories_registered(self) -> bool:
        """是否已注册任何工厂 (deps.py 惰性初始化用)。"""
        return len(self._factories) > 0

    def get(
        self,
        service_type: ServiceType,
        default: ServiceFactory | None = None,
    ) -> Service:
        """获取服务实例——首次调用 factory.create() 并缓存, 后续直接返回。

        Args:
            service_type: 服务类型枚举。
            default: 如果该类型工厂未注册, 用此默认工厂创建。

        Raises:
            ValueError: 既无注册工厂也无 default。
        """
        if service_type in self._services:
            return self._services[service_type]

        factory = self._factories.get(service_type)
        if factory is None:
            if default is not None:
                factory = default
            else:
                raise ValueError(f"No factory registered for {service_type}")

        service = factory.create()
        self._services[service_type] = service
        logger.info("service_created", service_type=service_type.value)
        return service

    @property
    def services(self) -> dict[ServiceType, Service]:
        """已创建的服务缓存。"""
        return self._services

    def is_initialized(self, service_type: ServiceType) -> bool:
        """检查服务是否已创建 (不触发创建)。"""
        return service_type in self._services

    async def teardown(self) -> None:
        """释放所有服务资源。"""
        for service in self._services.values():
            await service.teardown()
        self._services.clear()
        self._factories.clear()
        logger.info("service_manager_torn_down")


# ── 全局单例 ──

_global_manager: ServiceManager | None = None


def get_service_manager() -> ServiceManager:
    """获取全局 ServiceManager 单例。"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ServiceManager()
    return _global_manager
