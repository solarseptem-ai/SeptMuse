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
"""LoRAMemory 单元测试 (借鉴 HuggingFace PEFT adapter 生命周期管理 API 行为模型)。

覆盖:
- LoRAConfig dataclass 字段默认值 + asdict 转换
- LoRAMemory 初始化: 无参 / 仅 config / 仅 base_model / 同时给 (延迟 wrap)
- 无 base_model 时 save/load/set_active/merge_and_unload 抛 RuntimeError
- 无 peft 时 wrap 抛 ImportError 提示 (monkeypatch 模拟)
- adapter 元数据 CRUD: list_adapters / __len__ / __contains__
- dump / load_from 往返 (pickle 元数据 + config, 不含模型权重)
- peft/torch 真实环境集成 (skipif, 用 mock base_model 测试 wrap + save_pretrained 调用)
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock

import pytest

from septmuse.storage.file_stores import LoRAConfig, LoRAMemory
from septmuse.storage.file_stores.lora_base import BaseParametricMemory

# ======================================================================
# 环境检测 (用 skipif 控制粒度, 不让 module-level importorskip 跳过全文件)
# ======================================================================


_PEFT_AVAILABLE = True
_TORCH_AVAILABLE = True
_TRANSFORMERS_AVAILABLE = True
try:
    import peft as _peft  # noqa: F401
except ImportError:
    _PEFT_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False
try:
    import transformers as _transformers  # noqa: F401
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_NEEDS_PEFT_STACK = pytest.mark.skipif(
    not (_PEFT_AVAILABLE and _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE),
    reason="peft + torch + transformers 未安装 (需要 `pip install septmuse[parametric]`)",
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def lora_memory() -> Iterator[LoRAMemory]:
    """空 LoRAMemory (无 base_model, 无 config)。"""
    mem = LoRAMemory()
    yield mem


# ======================================================================
# LoRAConfig dataclass
# ======================================================================


class TestLoRAConfig:
    def test_default_r_is_8(self) -> None:
        cfg = LoRAConfig()
        assert cfg.r == 8

    def test_default_lora_alpha_is_16(self) -> None:
        cfg = LoRAConfig()
        assert cfg.lora_alpha == 16

    def test_default_lora_dropout_is_zero(self) -> None:
        cfg = LoRAConfig()
        assert cfg.lora_dropout == 0.0

    def test_default_target_modules_is_none(self) -> None:
        cfg = LoRAConfig()
        assert cfg.target_modules is None

    def test_default_bias_is_none_str(self) -> None:
        cfg = LoRAConfig()
        assert cfg.bias == "none"

    def test_default_task_type_is_causal_lm(self) -> None:
        cfg = LoRAConfig()
        assert cfg.task_type == "CAUSAL_LM"

    def test_custom_config(self) -> None:
        cfg = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="lora_only",
            task_type="SEQ_CLS",
        )
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.target_modules == ["q_proj", "v_proj"]
        assert cfg.bias == "lora_only"
        assert cfg.task_type == "SEQ_CLS"

    def test_asdict_roundtrip(self) -> None:
        cfg = LoRAConfig(r=64, lora_alpha=128, target_modules=["q_proj"])
        d = asdict(cfg)
        assert d["r"] == 64
        assert d["lora_alpha"] == 128
        assert d["target_modules"] == ["q_proj"]
        # 从 dict 重构
        cfg2 = LoRAConfig(**d)
        assert cfg2.r == cfg.r
        assert cfg2.lora_alpha == cfg.lora_alpha
        assert cfg2.target_modules == cfg.target_modules


# ======================================================================
# LoRAMemory 初始化 + ABC 继承
# ======================================================================


class TestLoRAMemoryInit:
    def test_inherits_base_parametric_memory(self, lora_memory: LoRAMemory) -> None:
        assert isinstance(lora_memory, BaseParametricMemory)

    def test_init_no_args(self, lora_memory: LoRAMemory) -> None:
        assert lora_memory.peft_model is None
        assert lora_memory.base_model is None
        assert lora_memory.config is None
        assert lora_memory.active_adapter is None
        assert lora_memory.list_adapters() == []
        assert len(lora_memory) == 0

    def test_init_only_config(self) -> None:
        cfg = LoRAConfig(r=32)
        mem = LoRAMemory(config=cfg)
        assert mem.config is cfg
        assert mem.peft_model is None  # 未 wrap
        assert len(mem) == 0

    def test_init_only_base_model(self) -> None:
        fake_model = object()
        mem = LoRAMemory(base_model=fake_model)
        assert mem.base_model is fake_model
        assert mem.peft_model is None  # 未 wrap
        assert len(mem) == 0

    def test_init_both_triggers_wrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """同时给 base_model + config 时立即 wrap (便捷初始化)。"""
        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")

        fake_model = object()
        cfg = LoRAConfig(r=16)

        # Mock get_peft_model 避免真实 PEFT 调用
        fake_peft_model = MagicMock(name="PeftModel")
        fake_get_peft_model = MagicMock(return_value=fake_peft_model)
        monkeypatch.setattr("peft.get_peft_model", fake_get_peft_model, raising=False)

        mem = LoRAMemory(base_model=fake_model, config=cfg)
        assert mem.peft_model is fake_peft_model
        assert "default" in mem  # wrap 后自动注册 default adapter
        assert mem.active_adapter == "default"


# ======================================================================
# 无 wrap 时的错误处理
# ======================================================================


class TestLoRAMemoryNoWrapErrors:
    def test_save_without_wrap_raises(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        with pytest.raises(RuntimeError, match="wrap"):
            lora_memory.save(str(tmp_path / "x"))

    def test_load_without_wrap_raises(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        with pytest.raises(RuntimeError, match="wrap"):
            lora_memory.load(str(tmp_path / "x"))

    def test_set_active_without_wrap_raises(self, lora_memory: LoRAMemory) -> None:
        with pytest.raises(RuntimeError, match="wrap"):
            lora_memory.set_active("default")

    def test_merge_and_unload_without_wrap_raises(self, lora_memory: LoRAMemory) -> None:
        with pytest.raises(RuntimeError, match="wrap"):
            lora_memory.merge_and_unload()

    def test_set_active_unknown_adapter_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """wrap 后但 adapter 未加载时报错。"""
        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")

        fake_peft_model = MagicMock(name="PeftModel")
        fake_get_peft_model = MagicMock(return_value=fake_peft_model)
        monkeypatch.setattr("peft.get_peft_model", fake_get_peft_model, raising=False)

        mem = LoRAMemory(base_model=object(), config=LoRAConfig())
        with pytest.raises(RuntimeError, match="not loaded"):
            mem.set_active("nonexistent")


# ======================================================================
# 无 peft/torch 时 wrap 抛 ImportError 提示 (monkeypatch 模拟)
# ======================================================================


class TestLoRAMemoryImportError:
    def test_wrap_without_peft_raises_import_error(
        self, lora_memory: LoRAMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模拟无 peft 环境, wrap 抛清晰 ImportError。"""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "peft":
                raise ImportError("simulated missing peft")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match="parametric"):
            lora_memory.wrap(object(), LoRAConfig())


# ======================================================================
# 真实 PEFT 环境集成 (skipif)
# ======================================================================


@_NEEDS_PEFT_STACK
class TestLoRAMemoryRealPEFT:
    """需要 peft + torch + transformers 真实环境 (CI 安装 parametric extras 时跑)。"""

    def test_wrap_creates_peft_model(self) -> None:
        """用最小 mock base_model 测试 wrap 返回 PeftModel-like 对象。

        注: 真实 PEFT get_peft_model 需要真实 transformers 模型;
        此测试用 MagicMock 模拟 base_model, 验证 wrap 调用 get_peft_model 而非模型语义。
        """
        from unittest.mock import patch

        fake_peft_model = MagicMock(name="PeftModel")
        with patch("peft.get_peft_model", return_value=fake_peft_model) as mock_get:
            mem = LoRAMemory()
            result = mem.wrap(MagicMock(name="base_model"), LoRAConfig(r=8, lora_alpha=16))
            assert result is fake_peft_model
            mock_get.assert_called_once()
            assert mem.peft_model is fake_peft_model
            assert "default" in mem
            assert mem.active_adapter == "default"

    def test_save_calls_save_pretrained(self, tmp_path: Any) -> None:
        """save 调用 peft_model.save_pretrained 并更新 adapter 元数据。"""
        from unittest.mock import patch

        fake_peft_model = MagicMock(name="PeftModel")
        with patch("peft.get_peft_model", return_value=fake_peft_model):
            mem = LoRAMemory()
            mem.wrap(MagicMock(name="base_model"), LoRAConfig())

        path = str(tmp_path / "adapters" / "v1")
        mem.save(path, adapter_name="default")
        fake_peft_model.save_pretrained.assert_called_once_with(path)
        assert mem.list_adapters() == ["default"]

    def test_load_calls_load_adapter(self, tmp_path: Any) -> None:
        """load 调用 peft_model.load_adapter。"""
        from unittest.mock import patch

        fake_peft_model = MagicMock(name="PeftModel")
        with patch("peft.get_peft_model", return_value=fake_peft_model):
            mem = LoRAMemory()
            mem.wrap(MagicMock(name="base_model"), LoRAConfig())

        path = str(tmp_path / "adapters" / "v2")
        mem.load(path, adapter_name="v2")
        fake_peft_model.load_adapter.assert_called_once()
        assert "v2" in mem

    def test_set_active_calls_set_adapter(self) -> None:
        """set_active 调用 peft_model.set_adapter。"""
        from unittest.mock import patch

        fake_peft_model = MagicMock(name="PeftModel")
        with patch("peft.get_peft_model", return_value=fake_peft_model):
            mem = LoRAMemory()
            mem.wrap(MagicMock(name="base_model"), LoRAConfig())

        # 先注册 v2 adapter 元数据 (绕过真实 load_adapter)
        from septmuse.storage.file_stores.lora import _AdapterState

        mem._adapters["v2"] = _AdapterState(name="v2", path="/fake/v2")
        mem.set_active("v2")
        fake_peft_model.set_adapter.assert_called_once_with("v2")
        assert mem.active_adapter == "v2"

    def test_merge_and_unload_calls_peft_method(self) -> None:
        """merge_and_unload 调用 peft_model.merge_and_unload, 返回 merged model。"""
        from unittest.mock import patch

        fake_peft_model = MagicMock(name="PeftModel")
        fake_merged = MagicMock(name="merged_model")
        fake_peft_model.merge_and_unload = MagicMock(return_value=fake_merged)
        with patch("peft.get_peft_model", return_value=fake_peft_model):
            mem = LoRAMemory()
            mem.wrap(MagicMock(name="base_model"), LoRAConfig())

        result = mem.merge_and_unload()
        assert result is fake_merged
        fake_peft_model.merge_and_unload.assert_called_once()
        # 合并后状态清空
        assert mem.peft_model is None
        assert len(mem) == 0
        assert mem.active_adapter is None


# ======================================================================
# dump / load_from 往返
# ======================================================================


class TestLoRAMemoryDumpLoad:
    def test_dump_creates_file(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        lora_memory._config = LoRAConfig(r=64)
        path = str(tmp_path / "lora.pkl")
        lora_memory.dump(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_load_from_nonexistent_raises(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        """load_from 文件不存在抛 FileNotFoundError (与 ActivationMemory 不同, 严格报错)。"""
        path = str(tmp_path / "nonexistent.pkl")
        with pytest.raises(FileNotFoundError):
            lora_memory.load_from(path)

    def test_dump_load_roundtrip(self, tmp_path: Any) -> None:
        """dump -> load_from 往返, config + adapter 元数据还原。"""
        original = LoRAMemory()
        original._config = LoRAConfig(r=64, lora_alpha=128, target_modules=["q_proj"])
        # 手动注册 adapter 元数据 (不经 wrap, 测试纯序列化)
        from septmuse.storage.file_stores.lora import _AdapterState

        original._adapters = {
            "v1": _AdapterState(name="v1", path="/fake/v1"),
            "v2": _AdapterState(name="v2", path="/fake/v2"),
        }
        original._active_adapter = "v1"

        path = str(tmp_path / "roundtrip.pkl")
        original.dump(path)

        loaded = LoRAMemory()
        loaded.load_from(path)
        assert loaded.config is not None
        assert loaded.config.r == 64
        assert loaded.config.lora_alpha == 128
        assert loaded.config.target_modules == ["q_proj"]
        assert set(loaded.list_adapters()) == {"v1", "v2"}
        assert loaded.active_adapter == "v1"
        # _peft_model 不还原 (模型不可 pickle)
        assert loaded.peft_model is None

    def test_dump_creates_parent_dir(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        """dump 自动创建父目录。"""
        lora_memory._config = LoRAConfig()
        path = str(tmp_path / "subdir" / "nested" / "lora.pkl")
        lora_memory.dump(path)
        assert os.path.exists(path)

    def test_load_from_corrupted_raises(self, lora_memory: LoRAMemory, tmp_path: Any) -> None:
        """损坏文件抛 (EOFError / UnpicklingError)。"""
        path = str(tmp_path / "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"not a pickle")

        # load_from 捕获 EOFError / pickle.UnpicklingError 后 raise; 这里验证不静默吞异常
        with pytest.raises((EOFError, pickle.UnpicklingError)):
            lora_memory.load_from(path)


# ======================================================================
# 独立工具: 模块零强依赖
# ======================================================================


class TestLoRAMemoryStandalone:
    def test_can_import_without_peft_installed(self) -> None:
        """模块本身不依赖 peft/torch, 无 parametric extras 也能 import。"""
        from septmuse.storage.file_stores import LoRAConfig, LoRAMemory

        assert LoRAMemory is not None
        assert LoRAConfig is not None

    def test_config_crud_works_without_peft(self) -> None:
        """无 peft 时 LoRAConfig + 内存元数据 CRUD 仍可用 (只 wrap/save/load 报错)。"""
        mem = LoRAMemory()
        mem._config = LoRAConfig(r=32)
        cfg = mem.config
        assert cfg is not None
        assert cfg.r == 32
        assert len(mem) == 0
        assert mem.list_adapters() == []
        assert mem.active_adapter is None
