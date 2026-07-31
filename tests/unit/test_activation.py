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
"""ActivationMemory 单元测试 (借鉴 MemOS KVCacheMemory 行为模型)。

覆盖:
- KVCacheItem pydantic 字段默认值 + arbitrary_types_allowed
- ActivationMemory CRUD: add / get / get_by_ids / get_all / delete / delete_all
- get_cache 合并: 空 / 单个 / 多个 (用 fake cache 对象, 不依赖真实 DynamicCache)
- extract: 注入 cache_builder 回调; 无 builder 抛 NotImplementedError
- dump / load 往返 (用 pickle 友好的 fake cache 类)
- _concat_caches ImportError 提示 (无 torch/transformers 时, 用 monkeypatch 模拟)
- __len__ / __contains__
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from septmuse.storage.activation import ActivationMemory, KVCacheItem

# ======================================================================
# Fake cache 对象 (不依赖 torch / transformers, 用于单元测试 _concat_caches 之外的路径)
# ======================================================================


class FakeCache:
    """pickle 友好的简单 cache 替身 (用于 dump/load 往返测试)。"""

    def __init__(self, payload: str = "") -> None:
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeCache) and self.payload == other.payload


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def activation() -> Iterator[ActivationMemory]:
    """空 ActivationMemory (无 cache_builder)。"""
    mem = ActivationMemory()
    yield mem
    mem.delete_all()


@pytest.fixture()
def activation_with_builder() -> Iterator[ActivationMemory]:
    """带 fake cache_builder 的 ActivationMemory。"""

    def _builder(text: str) -> FakeCache:
        return FakeCache(payload=text)

    mem = ActivationMemory(cache_builder=_builder)
    yield mem
    mem.delete_all()


# ======================================================================
# KVCacheItem pydantic model
# ======================================================================


class TestKVCacheItem:
    def test_default_id_is_uuid_str(self) -> None:
        item = KVCacheItem()
        assert isinstance(item.id, str)
        # uuid 字符串长度 36
        assert len(item.id) == 36

    def test_default_memory_is_none(self) -> None:
        item = KVCacheItem()
        assert item.memory is None

    def test_default_metadata_is_empty_dict(self) -> None:
        item = KVCacheItem()
        assert item.metadata == {}

    def test_arbitrary_types_allowed(self) -> None:
        """memory 字段可持有任意类型 (DynamicCache / FakeCache / dict 等)。"""
        cache = FakeCache("hello")
        item = KVCacheItem(memory=cache, metadata={"source": "test"})
        assert item.memory is cache
        assert item.metadata["source"] == "test"

    def test_explicit_id(self) -> None:
        item = KVCacheItem(id="custom-id-123")
        assert item.id == "custom-id-123"

    def test_unique_ids(self) -> None:
        """两个默认 item 的 id 不同。"""
        a = KVCacheItem()
        b = KVCacheItem()
        assert a.id != b.id


# ======================================================================
# ActivationMemory.__init__
# ======================================================================


class TestActivationMemoryInit:
    def test_init_no_builder(self) -> None:
        mem = ActivationMemory()
        assert mem._cache_builder is None
        assert len(mem) == 0

    def test_init_with_builder(self) -> None:
        def builder(text: str) -> Any:
            return FakeCache(text)

        mem = ActivationMemory(cache_builder=builder)
        assert mem._cache_builder is builder

    def test_init_empty_items(self) -> None:
        mem = ActivationMemory()
        assert mem.get_all() == []


# ======================================================================
# add / get / get_by_ids / get_all / delete / delete_all
# ======================================================================


class TestActivationMemoryCRUD:
    def test_add_and_get(self, activation: ActivationMemory) -> None:
        item = KVCacheItem(memory=FakeCache("a"), metadata={"k": "v"})
        activation.add(item)
        assert activation.get(item.id) is item

    def test_add_idempotent_by_id(self, activation: ActivationMemory) -> None:
        """同 id 再 add 覆盖原 item (dict 语义)。"""
        item1 = KVCacheItem(id="x", memory=FakeCache("v1"))
        activation.add(item1)
        item2 = KVCacheItem(id="x", memory=FakeCache("v2"))
        activation.add(item2)
        assert len(activation) == 1
        got = activation.get("x")
        assert got is not None
        assert got.memory.payload == "v2"

    def test_get_missing_returns_none(self, activation: ActivationMemory) -> None:
        assert activation.get("nonexistent") is None

    def test_get_by_ids(self, activation: ActivationMemory) -> None:
        a = KVCacheItem(memory=FakeCache("a"))
        b = KVCacheItem(memory=FakeCache("b"))
        activation.add(a)
        activation.add(b)
        result = activation.get_by_ids([a.id, "missing", b.id])
        assert result[0] is a
        assert result[1] is None
        assert result[2] is b

    def test_get_by_ids_empty(self, activation: ActivationMemory) -> None:
        assert activation.get_by_ids([]) == []

    def test_get_all(self, activation: ActivationMemory) -> None:
        a = KVCacheItem(memory=FakeCache("a"))
        b = KVCacheItem(memory=FakeCache("b"))
        activation.add(a)
        activation.add(b)
        all_items = activation.get_all()
        assert len(all_items) == 2
        assert a in all_items
        assert b in all_items

    def test_get_all_empty(self, activation: ActivationMemory) -> None:
        assert activation.get_all() == []

    def test_delete_single(self, activation: ActivationMemory) -> None:
        a = KVCacheItem(memory=FakeCache("a"))
        activation.add(a)
        activation.delete([a.id])
        assert activation.get(a.id) is None
        assert len(activation) == 0

    def test_delete_multiple(self, activation: ActivationMemory) -> None:
        a = KVCacheItem(memory=FakeCache("a"))
        b = KVCacheItem(memory=FakeCache("b"))
        c = KVCacheItem(memory=FakeCache("c"))
        activation.add(a)
        activation.add(b)
        activation.add(c)
        activation.delete([a.id, c.id])
        assert activation.get(a.id) is None
        assert activation.get(b.id) is b
        assert activation.get(c.id) is None

    def test_delete_missing_silent(self, activation: ActivationMemory) -> None:
        """删除不存在的 id 静默忽略。"""
        activation.delete(["nonexistent"])  # 不抛异常

    def test_delete_all(self, activation: ActivationMemory) -> None:
        activation.add(KVCacheItem(memory=FakeCache("a")))
        activation.add(KVCacheItem(memory=FakeCache("b")))
        activation.delete_all()
        assert len(activation) == 0
        assert activation.get_all() == []


# ======================================================================
# __len__ / __contains__
# ======================================================================


class TestActivationMemoryDunders:
    def test_len_empty(self, activation: ActivationMemory) -> None:
        assert len(activation) == 0

    def test_len_after_add(self, activation: ActivationMemory) -> None:
        activation.add(KVCacheItem(memory=FakeCache("a")))
        activation.add(KVCacheItem(memory=FakeCache("b")))
        assert len(activation) == 2

    def test_contains(self, activation: ActivationMemory) -> None:
        item = KVCacheItem(memory=FakeCache("a"))
        activation.add(item)
        assert item.id in activation
        assert "nonexistent" not in activation


# ======================================================================
# extract (依赖反转)
# ======================================================================


class TestActivationMemoryExtract:
    def test_extract_with_builder(self, activation_with_builder: ActivationMemory) -> None:
        item = activation_with_builder.extract("固定背景文本")
        assert isinstance(item, KVCacheItem)
        assert item.memory.payload == "固定背景文本"
        assert item.metadata["source_text"] == "固定背景文本"
        assert "extracted_at" in item.metadata

    def test_extract_without_builder_raises(self, activation: ActivationMemory) -> None:
        with pytest.raises(NotImplementedError, match="cache_builder"):
            activation.extract("text")

    def test_extract_metadata_has_iso_timestamp(self, activation_with_builder: ActivationMemory) -> None:
        item = activation_with_builder.extract("text")
        # ISO 8601 含 'T'
        assert "T" in item.metadata["extracted_at"]


# ======================================================================
# get_cache (合并)
# ======================================================================


class TestActivationMemoryGetCache:
    def test_get_cache_empty_ids_returns_none(self, activation: ActivationMemory) -> None:
        assert activation.get_cache([]) is None

    def test_get_cache_missing_ids_returns_none(self, activation: ActivationMemory) -> None:
        assert activation.get_cache(["nonexistent"]) is None

    def test_get_cache_single_returns_memory_directly(self, activation: ActivationMemory) -> None:
        """单个 item 时直接返回其 memory, 不走 _concat_caches。"""
        cache = FakeCache("only")
        item = KVCacheItem(memory=cache)
        activation.add(item)
        result = activation.get_cache([item.id])
        assert result is cache

    def test_get_cache_skips_items_with_none_memory(self, activation: ActivationMemory) -> None:
        """memory=None 的 item 被跳过 (借鉴 MemOS if item and item.memory)。"""
        empty_item = KVCacheItem(memory=None)
        cache_item = KVCacheItem(memory=FakeCache("real"))
        activation.add(empty_item)
        activation.add(cache_item)
        # 只有 1 个非 None -> 直接返回, 不走 _concat_caches
        result = activation.get_cache([empty_item.id, cache_item.id])
        assert isinstance(result, FakeCache)
        assert result.payload == "real"

    def test_get_cache_multiple_with_fake_caches_raises_or_returns(
        self, activation: ActivationMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """多个 item 时走 _concat_caches, 但 FakeCache 不兼容 torch.cat 流程。

        验证: 若环境无 torch/transformers, 抛 ImportError; 否则 _concat_caches 走到
        `DynamicCache object has neither 'layers' nor 'key_cache'` 分支。
        """
        a = KVCacheItem(memory=FakeCache("a"))
        b = KVCacheItem(memory=FakeCache("b"))
        activation.add(a)
        activation.add(b)

        # 模拟无 torch/transformers 环境
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name == "transformers":
                raise ImportError(f"simulated missing: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match="activation"):
            activation.get_cache([a.id, b.id])


# ======================================================================
# _concat_caches 直接测试 (有 torch/transformers 时用真实 DynamicCache, 否则跳过)
# ======================================================================


# 检测 torch + transformers 可用性 (不抛 Skipped, 用 skipif marker 控制粒度)
_TORCH_AVAILABLE = True
_TRANSFORMERS_AVAILABLE = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False
try:
    import transformers as _transformers  # noqa: F401
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_NEEDS_TORCH = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE),
    reason="torch + transformers 未安装 (需要 `pip install septmuse[activation]`)",
)


@_NEEDS_TORCH
class TestConcatCachesReal:
    """需要 torch + transformers 真实环境 (CI 安装 activation extras 时跑)。"""

    def test_concat_two_caches_layers_structure(self) -> None:
        """新版 transformers DynamicCache (layers[i].keys / layers[i].values) 合并。"""
        import torch
        from transformers import DynamicCache

        c1 = DynamicCache()
        c2 = DynamicCache()
        # 新版接口: 用 update(k, v, layer_idx=N) 初始化 layer
        c1.update(torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2, 8), layer_idx=0)
        c2.update(torch.ones(1, 4, 3, 8), torch.ones(1, 4, 3, 8), layer_idx=0)

        mem = ActivationMemory()
        merged = mem._concat_caches([c1, c2])
        assert isinstance(merged, DynamicCache)
        # 合并后 seq_len = 2 + 3 = 5
        assert merged.layers[0].keys is not None
        assert merged.layers[0].values is not None
        assert merged.layers[0].keys.shape[-2] == 5
        assert merged.layers[0].values.shape[-2] == 5

    def test_concat_single_passthrough_in_get_cache(self) -> None:
        """get_cache 单 item 直接返回 memory (不走 _concat_caches)。"""
        import torch
        from transformers import DynamicCache

        cache = DynamicCache()
        cache.update(torch.zeros(1, 4, 5, 8), torch.zeros(1, 4, 5, 8), layer_idx=0)

        mem = ActivationMemory()
        item = KVCacheItem(memory=cache)
        mem.add(item)
        result = mem.get_cache([item.id])
        assert result is cache


# ======================================================================
# dump / load 往返
# ======================================================================


class TestActivationMemoryDumpLoad:
    def test_dump_creates_file(self, activation: ActivationMemory, tmp_path: Any) -> None:
        """dump 需 torch/transformers, 没装则跳过。"""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        activation.add(KVCacheItem(memory=FakeCache("a"), metadata={"k": "v"}))
        path = str(tmp_path / "act.pkl")
        activation.dump(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_load_nonexistent_file_silent(self, activation: ActivationMemory, tmp_path: Any) -> None:
        """文件不存在 -> 静默返回 (保持空状态)。"""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        path = str(tmp_path / "nonexistent.pkl")
        activation.load(path)  # 不抛异常
        assert len(activation) == 0

    def test_dump_load_roundtrip(self, tmp_path: Any) -> None:
        """dump -> load 往返, items 还原。"""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        # 原 mem
        original = ActivationMemory()
        a = KVCacheItem(memory=FakeCache("payload-a"), metadata={"k": "v"})
        b = KVCacheItem(memory=FakeCache("payload-b"))
        original.add(a)
        original.add(b)

        path = str(tmp_path / "roundtrip.pkl")
        original.dump(path)

        # 新 mem load
        loaded = ActivationMemory()
        loaded.load(path)
        assert len(loaded) == 2
        got_a = loaded.get(a.id)
        assert got_a is not None
        assert got_a.memory == FakeCache("payload-a")
        assert got_a.metadata["k"] == "v"
        got_b = loaded.get(b.id)
        assert got_b is not None
        assert got_b.memory == FakeCache("payload-b")

    def test_load_corrupted_file_resets_to_empty(self, activation: ActivationMemory, tmp_path: Any) -> None:
        """损坏文件 -> 重置为空 (借鉴 MemOS 异常处理)。"""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        path = str(tmp_path / "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"not a pickle")

        activation.add(KVCacheItem(memory=FakeCache("preexisting")))
        activation.load(path)
        assert len(activation) == 0  # 重置为空

    def test_dump_creates_parent_dir(self, activation: ActivationMemory, tmp_path: Any) -> None:
        """dump 自动创建父目录 (os.makedirs exist_ok=True)。"""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        activation.add(KVCacheItem(memory=FakeCache("a")))
        path = str(tmp_path / "subdir" / "nested" / "act.pkl")
        activation.dump(path)
        assert os.path.exists(path)


# ======================================================================
# 集成: Memory facade 是否应该暴露 activation (架构文档 §11.2 未列端点, §9 阶段5 标可选)
# ======================================================================


class TestActivationMemoryStandalone:
    """activation 是独立工具, 不进 Memory facade (避免零配置场景强依赖 torch)。"""

    def test_can_import_without_torch_installed(self) -> None:
        """模块本身不依赖 torch, 无 activation extras 也能 import。"""
        from septmuse.storage.activation import ActivationMemory, KVCacheItem

        assert ActivationMemory is not None
        assert KVCacheItem is not None

    def test_crud_works_without_torch(self) -> None:
        """无 torch/transformers 时 CRUD 仍可用 (只 _concat_caches/dump/load 报错)。"""
        mem = ActivationMemory()
        item = KVCacheItem(memory=FakeCache("a"))
        mem.add(item)
        assert mem.get(item.id) is item
        assert len(mem) == 1
        mem.delete([item.id])
        assert len(mem) == 0
