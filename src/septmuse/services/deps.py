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
"""依赖注入入口 (借鉴 Langflow services/deps.py)。

这是 SeptMuse 服务层的**唯一对外入口**——每个服务一个 get_xxx_service() 函数,
内部走 ServiceManager.get(ServiceType.XXX, default_factory)。

用法:
    from septmuse.services.deps import get_memory_service

    memory = get_memory_service()
    memory.add("hello", user_id="alice")

模式 (对齐 Langflow deps.py):
    def get_xxx_service() -> XxxService:
        from septmuse.services.xxx.factory import XxxServiceFactory
        return get_service(ServiceType.XXX_SERVICE, XxxServiceFactory())

每个 get 函数:
    1. lazy import 对应 factory (避免循环依赖)
    2. 传 default factory 给 get_service (未注册时自动创建)
    3. ServiceManager 首次调用 create() 并缓存, 后续直接返回单例
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from septmuse.services.factory import ServiceFactory
from septmuse.services.schema import ServiceType

if TYPE_CHECKING:
    from septmuse.services.base import Service
    from septmuse.services.config.service import ConfigService
    from septmuse.services.embedder.service import EmbedderService
    from septmuse.services.llm.service import LLMService


def get_service(
    service_type: ServiceType,
    default: type[ServiceFactory] | None = None,
) -> Service:
    """通用服务获取——ServiceManager 单例 + default factory 惰性创建。

    借鉴 Langflow deps.py get_service():
    - 首次调用: manager 未注册 → 用 default factory 创建并缓存
    - 后续调用: 直接返回缓存的单例

    Args:
        service_type: ServiceType 枚举。
        default: 默认 ServiceFactory 子类 (未注册时自动实例化并创建)。
    """
    from septmuse.services.manager import get_service_manager

    manager = get_service_manager()

    if not manager.are_factories_registered():
        manager.register_factories(_get_default_factories())

    if default is not None and service_type not in manager._factories:
        manager.register_factory(service_type, default())

    return manager.get(service_type)


def is_service_initialized(service_type: ServiceType) -> bool:
    """检查服务是否已初始化 (不触发创建)。

    借鉴 Langflow is_settings_service_initialized。
    """
    from septmuse.services.manager import get_service_manager

    return service_type in get_service_manager().services


# ── 默认工厂注册表 ──


def _get_default_factories() -> dict[ServiceType, ServiceFactory]:
    """返回所有已实现的默认工厂 (惰性 import, 避免循环依赖)。

    每个具体服务实现后, 在此处注册对应 Factory 实例。
    未实现的跳过 (不报错, deps.py 的 get_xxx_service() 会传 default)。
    """
    factories: dict[ServiceType, ServiceFactory] = {}

    # ── P0: 配置 / 嵌入 / LLM ──
    from septmuse.services.config.factory import ConfigServiceFactory

    factories[ServiceType.CONFIG_SERVICE] = ConfigServiceFactory()

    from septmuse.services.embedder.factory import EmbedderServiceFactory

    factories[ServiceType.EMBEDDER_SERVICE] = EmbedderServiceFactory()

    from septmuse.services.llm.factory import LLMServiceFactory

    factories[ServiceType.LLM_SERVICE] = LLMServiceFactory()

    # ── P1 (待实现) ──
    # from septmuse.services.memory.factory import MemoryServiceFactory
    # factories[ServiceType.MEMORY_SERVICE] = MemoryServiceFactory()
    #
    # from septmuse.services.retrieval.factory import RetrievalServiceFactory
    # factories[ServiceType.RETRIEVAL_SERVICE] = RetrievalServiceFactory()
    #
    # from septmuse.services.ingestion.factory import IngestionServiceFactory
    # factories[ServiceType.INGESTION_SERVICE] = IngestionServiceFactory()
    #
    # ── P2 (待实现) ──
    # from septmuse.services.governance.factory import GovernanceServiceFactory
    # factories[ServiceType.GOVERNANCE_SERVICE] = GovernanceServiceFactory()

    return factories


# ── 对外暴露的 get_xxx_service() 函数 ──
# 每个服务一个函数, lazy import factory, 传 default 给 get_service。


def get_config_service() -> ConfigService:
    """获取配置服务 (包装 MemoryConfig / default_config)。"""
    from septmuse.services.config.factory import ConfigServiceFactory

    return typing.cast("ConfigService", get_service(ServiceType.CONFIG_SERVICE, ConfigServiceFactory))


def get_embedder_service() -> EmbedderService:
    """获取嵌入服务 (包装 Embedder)。"""
    from septmuse.services.embedder.factory import EmbedderServiceFactory

    return typing.cast("EmbedderService", get_service(ServiceType.EMBEDDER_SERVICE, EmbedderServiceFactory))


def get_llm_service() -> LLMService:
    """获取 LLM 服务 (包装 LLM provider)。"""
    from septmuse.services.llm.factory import LLMServiceFactory

    return typing.cast("LLMService", get_service(ServiceType.LLM_SERVICE, LLMServiceFactory))


# ── P1 (待实现) ──
# def get_memory_service() -> "MemoryService":
#     """获取记忆服务 (包装 Memory facade)。"""
#     from septmuse.services.memory.factory import MemoryServiceFactory
#     return get_service(ServiceType.MEMORY_SERVICE, MemoryServiceFactory)
#
# def get_retrieval_service() -> "RetrievalService":
#     """获取检索服务 (包装 HybridRetriever + Reranker)。"""
#     from septmuse.services.retrieval.factory import RetrievalServiceFactory
#     return get_service(ServiceType.RETRIEVAL_SERVICE, RetrievalServiceFactory)
#
# def get_ingestion_service() -> "IngestionService":
#     """获取摄入服务 (包装 FactExtractor + CognifyPipeline)。"""
#     from septmuse.services.ingestion.factory import IngestionServiceFactory
#     return get_service(ServiceType.INGESTION_SERVICE, IngestionServiceFactory)
#
# ── P2 (待实现) ──
# def get_governance_service() -> "GovernanceService":
#     """获取治理服务 (权限 + 审计 + 状态机)。"""
#     from septmuse.services.governance.factory import GovernanceServiceFactory
#     return get_service(ServiceType.GOVERNANCE_SERVICE, GovernanceServiceFactory)
