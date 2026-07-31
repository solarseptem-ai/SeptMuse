"""ServiceProvider — 能力后端容器, 按需 import + 实例化。

resolve() 延迟 import 后端模块, 仅在首次调用时加载。
is_available() 用 importlib.util.find_spec 检查依赖, 不 import 后端模块本身。
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Generic, TypeVar

from septmuse.services.registry import _DEFAULTS, BACKEND_MANIFEST, BackendEntry

T = TypeVar("T")


class ServiceProvider(Generic[T]):
    """能力后端容器。绑定一个能力, 持有该能力的 manifest 子集和代码默认值。

    所有实例化延迟到 resolve() 调用时: 先按 backend 查 manifest, 再 import 模块,
    最后用 config/kwargs 实例化。类对象缓存到 _class_cache 避免重复 import。
    """

    def __init__(self, capability: str, manifest: dict[str, BackendEntry], default: str) -> None:
        self._capability = capability
        self._manifest = manifest
        self._default = default
        self._class_cache: dict[str, type] = {}

    def resolve(self, backend: str | None = None, *, config: Any | None = None, **kwargs) -> T | None:
        """解析后端 -> 按需 import -> 实例化。

        backend=None 用代码默认; 空串返回 None (llm 默认不创建);
        none 后端 (cls 为空) 也返回 None; 未知 backend 抛 ValueError。
        """
        # 1. 确定后端名: None 用代码默认
        if backend is None:
            backend = self._default
        # 2. 空串 -> None (llm 默认不创建)
        if not backend:
            return None
        # 3. 未知后端 -> ValueError
        if backend not in self._manifest:
            raise ValueError(f"Unknown backend: {backend} for capability {self._capability}")
        entry = self._manifest[backend]
        # 4. none 后端 (cls 为空) -> None
        if not entry.cls:
            return None
        # 5. 按需 import 类
        cls = self._import_class(entry.module, entry.cls)
        # 6. 处理 config -> kwargs
        if config is not None:
            cfg_kwargs = self._config_to_kwargs(config, entry)
            cfg_kwargs.update(kwargs)
            kwargs = cfg_kwargs
        elif entry.config_cls is None and not kwargs:
            # search_recipe: get_recipe(name=...) 等无 config 类的函数式后端
            kwargs = {"name": backend}
        # 7. 实例化
        return cls(**kwargs)

    def list_backends(self) -> list[str]:
        """列出该能力所有注册的后端名。"""
        return list(self._manifest.keys())

    def is_available(self, backend: str) -> bool:
        """检查后端依赖是否已安装。用 find_spec 探测, 不 import 后端模块。"""
        if backend not in self._manifest:
            return False
        entry = self._manifest[backend]
        if not entry.deps:
            return True
        return all(importlib.util.find_spec(dep) is not None for dep in entry.deps)

    def available_backends(self) -> list[str]:
        """列出依赖已安装 (即可用) 的后端名。"""
        return [name for name in self._manifest if self.is_available(name)]

    def default_backend(self) -> str:
        """返回代码默认后端名。"""
        return self._default

    def _import_class(self, module_path: str, class_name: str) -> type:
        """按需 import 模块并 getattr 取类/函数, 缓存避免重复 import。"""
        cache_key = f"{module_path}.{class_name}"
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self._class_cache[cache_key] = cls
        return cls

    @staticmethod
    def _config_to_kwargs(config: Any, entry: BackendEntry) -> dict:
        """从 config 对象提取实例化参数，剥离 backend 元数据字段。"""
        if entry.config_cls is None:
            # 无 config 类: pydantic 对象走 model_dump, 否则当 name 传
            if hasattr(config, "model_dump"):
                result = config.model_dump()
            else:
                return {"name": config}
        elif hasattr(config, "model_dump"):
            result = config.model_dump(exclude_none=True)
        elif isinstance(config, dict):
            result = dict(config)
        else:
            return {}
        # backend 是配置元数据（标识用哪个后端），不是构造器参数，剥离
        result.pop("backend", None)
        return result


# 8 个能力的全局 provider 实例。
vector_store_provider: ServiceProvider = ServiceProvider(
    "vector_store", BACKEND_MANIFEST["vector_store"], _DEFAULTS["vector_store"]
)
embedder_provider: ServiceProvider = ServiceProvider(
    "embedder", BACKEND_MANIFEST["embedder"], _DEFAULTS["embedder"]
)
llm_provider: ServiceProvider = ServiceProvider(
    "llm", BACKEND_MANIFEST["llm"], _DEFAULTS["llm"]
)
reranker_provider: ServiceProvider = ServiceProvider(
    "reranker", BACKEND_MANIFEST["reranker"], _DEFAULTS["reranker"]
)
entity_extractor_provider: ServiceProvider = ServiceProvider(
    "entity_extractor", BACKEND_MANIFEST["entity_extractor"], _DEFAULTS["entity_extractor"]
)
keyword_index_provider: ServiceProvider = ServiceProvider(
    "keyword_index", BACKEND_MANIFEST["keyword_index"], _DEFAULTS["keyword_index"]
)
graph_store_provider: ServiceProvider = ServiceProvider(
    "graph_store", BACKEND_MANIFEST["graph_store"], _DEFAULTS["graph_store"]
)
search_recipe_provider: ServiceProvider = ServiceProvider(
    "search_recipe", BACKEND_MANIFEST["search_recipe"], _DEFAULTS["search_recipe"]
)

# 能力名 -> provider 的统一索引, 供运行时按能力名查找。
ALL_PROVIDERS: dict[str, ServiceProvider] = {
    "vector_store": vector_store_provider,
    "embedder": embedder_provider,
    "llm": llm_provider,
    "reranker": reranker_provider,
    "entity_extractor": entity_extractor_provider,
    "keyword_index": keyword_index_provider,
    "graph_store": graph_store_provider,
    "search_recipe": search_recipe_provider,
}
