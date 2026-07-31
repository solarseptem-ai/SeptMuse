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
"""激活记忆 (KV-Cache) — 平面 B 存储形态之一 (借鉴 MemOS KVCacheMemory, 架构文档 §3.1.2/§4)。

仅当使用自托管模型 (HuggingFace 后端) 时启用; API/闭源模型跳过此格。
用途: 固定背景 / FAQ / 会话历史复用, 降低 TTFT (prefill skip)。

接口对齐 MemOS `memos.memories.activation.kv.KVCacheMemory`:
- extract(text)  -> KVCacheItem      (需注入 cache_builder 回调, 解耦 LLM ABC)
- add(item)                         (单条, 幂等按 id 去重)
- get(id) -> KVCacheItem | None
- get_by_ids(ids) -> list[KVCacheItem | None]
- get_all() -> list[KVCacheItem]
- delete(ids)                      (按 id 删除)
- delete_all()                     (清空)
- get_cache(ids) -> DynamicCache | None  (合并多个 cache, 注入 attention 跳过 prefill)
- dump(path) / load(path)          (pickle 序列化, 含 DynamicCache safe_globals)

依赖反转 (SeptMuse 适配):
- 不耦合 LLM ABC (SeptMuse LLM 仅有 complete 方法, 无 build_kv_cache)
- 调用方注入 `cache_builder: Callable[[str], Any]`, 自行用 transformers + 自托管模型前向计算
- _concat_caches / dump / load 延迟 import torch/transformers/pickle, 未装 activation extras 报清晰错误

参考源: opensource/MemOS/src/memos/memories/activation/kv.py
        opensource/MemOS/src/memos/memories/activation/item.py
"""

from __future__ import annotations

import os
import pickle
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KVCacheItem(BaseModel):
    """单个 KV cache 条目 (对齐 MemOS `KVCacheItem`)。

    `memory` 字段类型标为 `Any` 避免强依赖 transformers.DynamicCache;
    实际运行时若启用 activation extras, 该字段持有 DynamicCache 实例。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory: Any = Field(default=None, description="DynamicCache 实例 (或调用方自定义 cache 对象)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _missing_activation_deps() -> ImportError:
    """构造清晰错误, 提示安装 activation extras。"""
    return ImportError(
        "activation memory requires `pip install septmuse[activation]` "
        "(transformers >= 4.40, torch >= 2.2) for DynamicCache concat / pickle"
    )


class ActivationMemory:
    """KV-Cache 激活记忆 (借鉴 MemOS KVCacheMemory, 架构文档 §3.1.2)。

    零外部依赖 (无 torch/transformers 也能 import):
    - 内存 dict 持有 KVCacheItem, CRUD 全在内存
    - 仅 `_concat_caches` / `dump` / `load` 需要 torch/pickle, 延迟 import + 报清晰错误
    - `extract` 需调用方注入 `cache_builder: Callable[[str], Any]`

    用法 (自托管模型场景):

        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        model = AutoModelForCausalLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)

        def build_cache(text: str) -> DynamicCache:
            inputs = tokenizer(text, return_tensors="pt")
            with model.forward(**inputs, use_cache=True) as out:
                return out.past_key_values

        activation = ActivationMemory(cache_builder=build_cache)
        item = activation.extract("固定背景文本")
        activation.add(item)
        merged = activation.get_cache([item.id])  # 注入 attention 跳过 prefill

    用法 (无 cache_builder, 直接 add 预构造 item):

        activation = ActivationMemory()
        item = KVCacheItem(memory=some_cache_obj, metadata={"source_text": "..."})
        activation.add(item)
    """

    def __init__(self, cache_builder: Callable[[str], Any] | None = None) -> None:
        """初始化激活记忆。

        Args:
            cache_builder: 文本 -> DynamicCache 的回调 (调用方负责用 transformers 前向计算)。
                           None 时 extract() 抛 NotImplementedError。
        """
        self._cache_builder = cache_builder
        self._items: dict[str, KVCacheItem] = {}
        logger.info("activation_memory_init", has_cache_builder=cache_builder is not None)

    # ------------------------------------------------------------------
    # extract (依赖反转, 借鉴 MemOS KVCacheMemory.extract)
    # ------------------------------------------------------------------

    def extract(self, text: str) -> KVCacheItem:
        """从文本提取 KV cache (借鉴 MemOS KVCacheMemory.extract)。

        需构造时注入 `cache_builder`; 否则抛 NotImplementedError 提示。

        Args:
            text: 输入文本 (固定背景 / FAQ / 会话历史)

        Returns:
            KVCacheItem, metadata 含 source_text + extracted_at
        """
        if self._cache_builder is None:
            raise NotImplementedError(
                "extract() requires a cache_builder; pass it to ActivationMemory(cache_builder=...)"
            )
        cache = self._cache_builder(text)
        item = KVCacheItem(
            memory=cache,
            metadata={"source_text": text, "extracted_at": _utcnow_iso()},
        )
        logger.info("activation_extract_done", item_id=item.id, text_len=len(text))
        return item

    # ------------------------------------------------------------------
    # CRUD (对齐 MemOS KVCacheMemory add/get/get_by_ids/get_all/delete/delete_all)
    # ------------------------------------------------------------------

    def add(self, item: KVCacheItem) -> None:
        """添加 KV cache 条目 (幂等, 按 id 去重)。"""
        self._items[item.id] = item

    def get(self, item_id: str) -> KVCacheItem | None:
        """取单条, 不存在返回 None。"""
        return self._items.get(item_id)

    def get_by_ids(self, item_ids: list[str]) -> list[KVCacheItem | None]:
        """按 id 批量取, 缺失位置返回 None。"""
        return [self.get(i) for i in item_ids]

    def get_all(self) -> list[KVCacheItem]:
        """列出全部。"""
        return list(self._items.values())

    def delete(self, item_ids: list[str]) -> None:
        """按 id 批量删除 (不存在静默忽略)。"""
        for i in item_ids:
            self._items.pop(i, None)

    def delete_all(self) -> None:
        """清空全部。"""
        self._items.clear()

    # ------------------------------------------------------------------
    # get_cache (合并, 借鉴 MemOS KVCacheMemory.get_cache + _concat_caches)
    # ------------------------------------------------------------------

    def get_cache(self, item_ids: list[str]) -> Any | None:
        """合并多个 KV cache 为单个 DynamicCache, 注入 attention 跳过 prefill。

        Args:
            item_ids: 要合并的 cache item ID 列表

        Returns:
            合并后的 DynamicCache; 无匹配 item 返回 None; 单 item 直接返回其 cache。
        """
        caches_to_merge: list[Any] = []
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item is not None and item.memory is not None:
                caches_to_merge.append(item.memory)

        if not caches_to_merge:
            return None
        if len(caches_to_merge) == 1:
            return caches_to_merge[0]
        return self._concat_caches(caches_to_merge)

    def _concat_caches(self, caches: list[Any]) -> Any:
        """多层 torch.cat 合并 DynamicCache (借鉴 MemOS KVCacheMemory._concat_caches)。

        兼容新旧 transformers 版本:
        - 新版: cache.layers[i].keys / cache.layers[i].values
        - 旧版: cache.key_cache[i] / cache.value_cache[i]

        依赖 torch + transformers; 未装 activation extras 报清晰错误。
        """
        assert caches, "need at least one cache to concat"
        try:
            import torch
            from transformers import DynamicCache
        except ImportError as e:
            raise _missing_activation_deps() from e

        merged = DynamicCache()

        # 新版 transformers: layers 属性
        if hasattr(caches[0], "layers"):
            num_layers = len(caches[0].layers)
            if not hasattr(merged, "layers"):
                merged.layers = []
            if num_layers > 0:
                layer_cls = type(caches[0].layers[0])
                while len(merged.layers) < num_layers:
                    merged.layers.append(layer_cls())
            for layer in range(num_layers):
                keys = [c.layers[layer].keys for c in caches]
                vals = [c.layers[layer].values for c in caches]
                merged.layers[layer].keys = torch.cat(keys, dim=-2)
                merged.layers[layer].values = torch.cat(vals, dim=-2)

        # 旧版 transformers: key_cache / value_cache 属性
        elif hasattr(caches[0], "key_cache"):
            num_layers = len(caches[0].key_cache)
            for layer in range(num_layers):
                keys = [c.key_cache[layer] for c in caches]
                vals = [c.value_cache[layer] for c in caches]
                merged.key_cache.append(torch.cat(keys, dim=-2))
                merged.value_cache.append(torch.cat(vals, dim=-2))

        else:
            raise AttributeError(
                "DynamicCache object has neither 'layers' nor 'key_cache' attributes; unsupported transformers version"
            )

        return merged

    # ------------------------------------------------------------------
    # dump / load (借鉴 MemOS KVCacheMemory.dump / load)
    # ------------------------------------------------------------------

    def dump(self, path: str) -> None:
        """pickle 序列化全部 items 到文件 (借鉴 MemOS KVCacheMemory.dump)。

        Args:
            path: 目标文件路径 (目录不存在则创建)

        Raises:
            ImportError: torch/transformers 未安装 (DynamicCache 需要安全反序列化注册)
        """
        try:
            import torch
            from transformers import DynamicCache
        except ImportError as e:
            raise _missing_activation_deps() from e

        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        # 注册 DynamicCache / KVCacheItem 为 safe globals (借鉴 MemOS load 实现)
        torch.serialization.add_safe_globals([DynamicCache, KVCacheItem])

        data = {"kv_cache_items": self._items}
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("activation_dump_done", path=path, count=len(self._items))

    def load(self, path: str) -> None:
        """从文件反序列化 items (借鉴 MemOS KVCacheMemory.load)。

        - 文件不存在: 静默返回 (保持空状态)
        - 加载失败: 重置为空 (借鉴 MemOS 异常处理)

        Args:
            path: 源文件路径
        """
        if not os.path.exists(path):
            return

        try:
            import torch
            from transformers import DynamicCache
        except ImportError as e:
            raise _missing_activation_deps() from e

        torch.serialization.add_safe_globals([DynamicCache, KVCacheItem])

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception):
            logger.warning("activation_load_failed_reset", path=path)
            self._items = {}
            return

        if isinstance(data, dict) and "kv_cache_items" in data:
            items = data["kv_cache_items"]
            if isinstance(items, dict):
                self._items = items
            elif isinstance(items, list):
                # 兼容旧 list 格式
                self._items = {item.id: item for item in items}
            else:
                self._items = {}
        elif isinstance(data, list):
            # 旧版兼容: list -> dict
            self._items = {item.id: item for item in data}
        else:
            self._items = {}

        logger.info("activation_load_done", path=path, count=len(self._items))

    # ------------------------------------------------------------------
    # 便捷工具
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items
