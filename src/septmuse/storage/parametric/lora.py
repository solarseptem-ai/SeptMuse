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
"""LoRA 参数化记忆 — 平面 B 存储形态之一 (架构文档 §4, 借鉴 HuggingFace PEFT 标准库 API)。

把"记忆"以 LoRA adapter 权重形式直接编入模型参数, 改变模型行为。
仅当使用自托管模型 (HuggingFace 后端) 时启用; API/闭源模型跳过此格。

接口对齐 HuggingFace PEFT 生命周期管理 API (借鉴源, 非自研):
- wrap(model, config)            -> PeftModel    (PEFT: get_peft_model)
- save(path, adapter_name)                       (PEFT: save_pretrained)
- load(path, adapter_name)                       (PEFT: PeftModel.load_adapter)
- set_active(adapter_name)                       (PEFT: PeftModel.set_adapter)
- list_adapters() -> list[str]                   (PEFT: peft_config 属性 keys)
- merge_and_unload() -> torch.nn.Module          (PEFT: PeftModel.merge_and_unload)
- dump(path) / load_from(path)                   (MemOS dump/load 接口形态)

依赖反转 (SeptMuse 适配, 对齐 ActivationMemory 模式):
- 不耦合具体模型类 (base_model: Any, 调用方注入 HuggingFace AutoModelForCausalLM 等)
- 不实现训练逻辑 — 训练由调用方用 transformers Trainer 完成 (PEFT 标准用法)
- 延迟 import peft/torch/transformers/pickle, 未装 parametric extras 报清晰错误

参考源:
- HuggingFace PEFT (https://github.com/huggingface/peft) — LoraConfig / get_peft_model / PeftModel 标准 API
- opensource/MemOS/src/memos/memories/parametric/lora.py — LoRAMemory 类形态 (本身是 placeholder, 仅借鉴接口形状)
- opensource/MemOS/src/memos/memories/parametric/base.py — BaseParaMemory ABC 形态
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.parametric.base import BaseParametricMemory

logger = get_logger(__name__)


# ======================================================================
# LoRAConfig (借鉴 PEFT LoraConfig 关键参数, 简化为 SeptMuse 所需子集)
# ======================================================================


@dataclass
class LoRAConfig:
    """LoRA 配置 (借鉴 PEFT `peft.LoraConfig` 关键参数子集)。

    PEFT LoraConfig 完整字段参见 https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
    SeptMuse 只暴露常用子集; 调用方需更细控制时, 可直接用 peft.LoraConfig 实例
    替换此 dataclass (LoRAMemory.wrap 接受 Any config)。

    Attributes:
        r: LoRA rank (默认 8, 越大表达能力越强但参数越多)
        lora_alpha: 缩放系数 (scaling = alpha / r, 借鉴 PEFT)
        lora_dropout: dropout 概率 (默认 0.0, 训练时正则)
        target_modules: 应用 LoRA 的模块名列表 (如 ["q_proj", "v_proj"]); None 时 PEFT 自动推断
        bias: "none" / "all" / "lora_only" (是否训练 bias)
        task_type: 任务类型 (CAUSAL_LM / SEQ_CLS / TOKEN_CLS 等, 对齐 PEFT TaskType 枚举值字符串)
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def _missing_parametric_deps() -> ImportError:
    """构造清晰错误, 提示安装 parametric extras。"""
    return ImportError(
        "parametric memory requires `pip install septmuse[parametric]` "
        "(peft >= 0.11, torch >= 2.2, transformers >= 4.40) for LoRA adapter lifecycle"
    )


def _to_peft_config(config: Any) -> Any:
    """把 SeptMuse LoRAConfig 转为 PEFT LoraConfig; 若已是 peft.LoraConfig 直接返回。"""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as e:
        raise _missing_parametric_deps() from e

    if isinstance(config, LoraConfig):
        return config

    if isinstance(config, LoRAConfig):
        task_type_str = config.task_type
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.CAUSAL_LM
        return LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias=config.bias,
            task_type=task_type,
        )

    raise TypeError(f"unsupported config type: {type(config).__name__}; expected LoRAConfig or peft.LoraConfig")


# ======================================================================
# LoRAMemory (借鉴 PEFT adapter 生命周期管理 API)
# ======================================================================


@dataclass
class _AdapterState:
    """单个 adapter 的元数据 (pickle 友好, 不含权重; 权重由 save/load 管理)。"""

    name: str
    path: str | None = None  # 已保存的目录路径 (None 表示仅内存, 未保存)


class LoRAMemory(BaseParametricMemory):
    """LoRA 参数化记忆 (借鉴 PEFT adapter 生命周期管理 API, 架构文档 §4)。

    零外部依赖 (无 peft/torch/transformers 也能 import):
    - 内存 dict 持有 _AdapterState (元数据), 权重由 PEFT 管理
    - wrap/save/load/set_active/merge_and_unload 延迟 import peft/torch, 报清晰错误
    - dump/load_from pickle 序列化元数据 + config, 不含模型权重

    用法 (自托管模型场景):

        from transformers import AutoModelForCausalLM
        from septmuse.storage.parametric import LoRAMemory, LoRAConfig

        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        lora = LoRAMemory()
        peft_model = lora.wrap(base_model, LoRAConfig(r=16, lora_alpha=32))

        # 调用方用 transformers Trainer 训练 peft_model...
        # (SeptMuse 不实现训练, 走 PEFT 标准路径)

        lora.save("./lora-adapters/v1", adapter_name="v1")
        lora.load("./lora-adapters/v2", adapter_name="v2")
        lora.set_active("v2")
        merged = lora.merge_and_unload()  # 推理优化: 合并到 base model

    用法 (无 base_model 时也能 CRUD adapter 元数据, 仅真实 PEFT 操作报错):

        lora = LoRAMemory()
        lora.save("./adapters/v1", adapter_name="v1")  # 报错: 需先 wrap
    """

    def __init__(
        self,
        base_model: Any | None = None,
        config: LoRAConfig | Any | None = None,
    ) -> None:
        """初始化 LoRA 记忆。

        Args:
            base_model: 自托管 HuggingFace 模型; None 时延后 wrap() 注入
            config: LoRAConfig 或 peft.LoraConfig; None 时延后 wrap() 注入
        """
        self._peft_model: Any | None = None
        self._base_model: Any | None = base_model
        self._config: LoRAConfig | Any | None = config
        self._adapters: dict[str, _AdapterState] = {}
        self._active_adapter: str | None = None

        # 若 base_model + config 同时给, 立即 wrap (便捷初始化)
        if base_model is not None and config is not None:
            self.wrap(base_model, config)
        else:
            logger.info(
                "lora_memory_init",
                has_base_model=base_model is not None,
                has_config=config is not None,
            )

    # ------------------------------------------------------------------
    # wrap (借鉴 PEFT get_peft_model)
    # ------------------------------------------------------------------

    def wrap(self, base_model: Any, config: Any) -> Any:
        """用 LoRA config 包装 base model, 返回 PeftModel (借鉴 PEFT get_peft_model)。

        Args:
            base_model: HuggingFace AutoModelForCausalLM 等自托管模型实例
            config: LoRAConfig 或 peft.LoraConfig

        Returns:
            peft.PeftModel — 调用方在其上用 transformers Trainer 训练

        Raises:
            ImportError: peft 未安装 (需 `pip install septmuse[parametric]`)
        """
        peft_config = _to_peft_config(config)
        try:
            from peft import get_peft_model
        except ImportError as e:
            raise _missing_parametric_deps() from e

        self._base_model = base_model
        self._config = config
        self._peft_model = get_peft_model(base_model, peft_config)

        # 默认 adapter 名 "default" (PEFT get_peft_model 自动创建)
        self._adapters["default"] = _AdapterState(name="default", path=None)
        self._active_adapter = "default"

        logger.info(
            "lora_wrap_done",
            adapter_count=len(self._adapters),
            config_r=getattr(config, "r", None),
        )
        return self._peft_model

    # ------------------------------------------------------------------
    # save / load (借鉴 PEFT save_pretrained + PeftModel.load_adapter)
    # ------------------------------------------------------------------

    def save(self, path: str, adapter_name: str = "default") -> None:
        """保存 adapter 权重到目录 (借鉴 PEFT save_pretrained)。

        Args:
            path: 目标目录 (不存在则创建)
            adapter_name: 要保存的 adapter 名 (默认 "default")

        Raises:
            RuntimeError: 未先 wrap() 注入 base model
            ImportError: peft 未安装
        """
        if self._peft_model is None:
            raise RuntimeError("LoRAMemory.save requires wrap() first; call wrap(base_model, config)")

        os.makedirs(path, exist_ok=True)
        # PEFT save_pretrained 总是保存 active adapter; 先 set_adapter 切换
        if adapter_name != self._active_adapter and adapter_name in self._adapters:
            self.set_active(adapter_name)

        self._peft_model.save_pretrained(path)
        self._adapters[adapter_name] = _AdapterState(name=adapter_name, path=path)
        logger.info("lora_save_done", path=path, adapter_name=adapter_name)

    def load(self, path: str, adapter_name: str = "default") -> None:
        """从目录加载 adapter (借鉴 PEFT PeftModel.load_adapter)。

        Args:
            path: 源目录 (含 adapter_model.safetensors / adapter_config.json)
            adapter_name: 加载后的 adapter 名 (已存在则覆盖)

        Raises:
            RuntimeError: 未先 wrap() 注入 base model
            ImportError: peft 未安装
        """
        if self._peft_model is None:
            raise RuntimeError("LoRAMemory.load requires wrap() first; call wrap(base_model, config)")

        try:
            # PeftModel.load_adapter 是实例方法
            self._peft_model.load_adapter(path, adapter_name=adapter_name)
        except AttributeError:
            # 某些 PEFT 版本 / 模型类型可能无 load_adapter, 回退到 PeftModel.from_pretrained
            try:
                from peft import PeftModel
            except ImportError as ie:
                raise _missing_parametric_deps() from ie
            self._peft_model = PeftModel.from_pretrained(self._base_model, path, adapter_name=adapter_name)

        self._adapters[adapter_name] = _AdapterState(name=adapter_name, path=path)
        logger.info("lora_load_done", path=path, adapter_name=adapter_name)

    # ------------------------------------------------------------------
    # set_active / list_adapters (借鉴 PEFT set_adapter + peft_config 属性)
    # ------------------------------------------------------------------

    def set_active(self, adapter_name: str) -> None:
        """切换激活的 adapter (借鉴 PEFT PeftModel.set_adapter)。

        Args:
            adapter_name: 要激活的 adapter 名

        Raises:
            RuntimeError: 未 wrap() 或 adapter 未加载
            ImportError: peft 未安装
        """
        if self._peft_model is None:
            raise RuntimeError("LoRAMemory.set_active requires wrap() first")
        if adapter_name not in self._adapters:
            raise RuntimeError(f"adapter '{adapter_name}' not loaded; call load(path, adapter_name=...) first")

        try:
            self._peft_model.set_adapter(adapter_name)
        except AttributeError as e:
            # 某些 PEFT 版本可能用 set_adapters (复数)
            if hasattr(self._peft_model, "set_adapters"):
                self._peft_model.set_adapters([adapter_name])
            else:
                raise e

        self._active_adapter = adapter_name
        logger.info("lora_set_active_done", adapter_name=adapter_name)

    def list_adapters(self) -> list[str]:
        """列出已加载的 adapter 名。"""
        return list(self._adapters.keys())

    # ------------------------------------------------------------------
    # merge_and_unload (借鉴 PEFT PeftModel.merge_and_unload)
    # ------------------------------------------------------------------

    def merge_and_unload(self) -> Any:
        """合并 LoRA 权重到 base model, 返回标准 PyTorch model (借鉴 PEFT merge_and_unload)。

        合并后 adapter 权重从内存移除, 推理时无 PEFT 开销。
        本 LoRAMemory 实例的 _peft_model 重置为 None, 调用方持有返回的 merged model。

        Returns:
            torch.nn.Module — 合并后的标准模型

        Raises:
            RuntimeError: 未 wrap()
            ImportError: peft/torch 未安装
        """
        if self._peft_model is None:
            raise RuntimeError("LoRAMemory.merge_and_unload requires wrap() first")

        try:
            merged = self._peft_model.merge_and_unload()
        except AttributeError as e:
            raise RuntimeError(f"peft_model has no merge_and_unload: {e}") from e

        logger.info(
            "lora_merge_done",
            active_adapter=self._active_adapter,
            adapter_count=len(self._adapters),
        )
        # 合并后 PeftModel 失效, 清空状态
        self._peft_model = None
        self._adapters.clear()
        self._active_adapter = None
        return merged

    # ------------------------------------------------------------------
    # dump / load_from (借鉴 MemOS LoRAMemory.dump/load 接口形态)
    # ------------------------------------------------------------------

    def dump(self, path: str) -> None:
        """pickle 序列化 memory 元数据 + config 到文件 (借鉴 MemOS dump 接口形态)。

        注意: 不含模型权重 / adapter 权重, 仅 config + adapter 元数据列表。
        调用方需另用 save() 保存权重, load_from() 后再 load() 恢复。

        Args:
            path: 目标文件路径 (目录不存在则创建)
        """
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        config_dict: dict[str, Any] | None = None
        if isinstance(self._config, LoRAConfig):
            config_dict = asdict(self._config)
        elif self._config is not None:
            # peft.LoraConfig 或其他 — 转为 dict (尽量保字段)
            config_dict = {"_type": type(self._config).__name__, **getattr(self._config, "__dict__", {})}

        data = {
            "config": config_dict,
            "adapters": self._adapters,
            "active_adapter": self._active_adapter,
            "has_peft_model": self._peft_model is not None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("lora_dump_done", path=path, adapter_count=len(self._adapters))

    def load_from(self, path: str) -> None:
        """从 pickle 反序列化 memory 状态 (借鉴 MemOS load 接口形态)。

        还原 config + adapter 元数据列表 + active_adapter。
        不还原 _peft_model (模型实例不可 pickle, 调用方需重新 wrap + load 各 adapter)。

        Args:
            path: 源文件路径

        Raises:
            FileNotFoundError: 文件不存在 (与 ActivationMemory 不同, 这里严格报错)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"LoRAMemory state file not found: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning("lora_load_from_failed", path=path, error=str(e))
            raise

        config_dict = data.get("config")
        if config_dict is not None and "_type" not in config_dict:
            self._config = LoRAConfig(**config_dict)
        else:
            self._config = None  # peft.LoraConfig 或其他, 无法纯 pickle 还原

        self._adapters = data.get("adapters", {})
        self._active_adapter = data.get("active_adapter")
        # _peft_model 不还原 (模型实例不可 pickle)
        self._peft_model = None

        logger.info(
            "lora_load_from_done",
            path=path,
            adapter_count=len(self._adapters),
            config_restored=self._config is not None,
        )

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def peft_model(self) -> Any | None:
        """当前 PeftModel 实例 (None 表示未 wrap 或已 merge)。"""
        return self._peft_model

    @property
    def base_model(self) -> Any | None:
        """原始 base model。"""
        return self._base_model

    @property
    def config(self) -> LoRAConfig | Any | None:
        """LoRA 配置。"""
        return self._config

    @property
    def active_adapter(self) -> str | None:
        """当前激活的 adapter 名。"""
        return self._active_adapter

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, adapter_name: str) -> bool:
        return adapter_name in self._adapters
