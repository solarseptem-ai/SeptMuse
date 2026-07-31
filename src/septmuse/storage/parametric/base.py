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
"""参数化记忆抽象基类 (借鉴 MemOS BaseParaMemory)。

参数化记忆把"记忆"以 LoRA / adapter 权重形式直接编入模型参数,
相比 KV-cache (激活记忆) 更持久, 训练时改变模型行为, 推理时无额外开销。

仅当使用自托管模型 (HuggingFace 后端) 时启用; API/闭源模型跳过此格。

借鉴源:
- opensource/MemOS/src/memos/memories/parametric/base.py (BaseParaMemory ABC 形态)
- HuggingFace PEFT (get_peft_model / PeftModel API 标准接口)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParametricMemory(ABC):
    """参数化记忆抽象 (借鉴 MemOS BaseParaMemory)。

    实现方需保证:
    - adapter 生命周期可独立管理 (保存/加载/切换/合并)
    - 不训练, 训练由调用方用 transformers Trainer 完成 (PEFT 标准用法)
    - 延迟 import peft/torch, 未装 parametric extras 报清晰错误
    """

    @abstractmethod
    def __init__(self, base_model: Any | None = None, config: Any | None = None) -> None:
        """初始化参数化记忆。

        Args:
            base_model: 自托管 HuggingFace 模型 (AutoModelForCausalLM 等); None 时延后注入
            config: LoRA 配置 (LoRAConfig 等); None 时延后注入
        """
        ...

    @abstractmethod
    def wrap(self, base_model: Any, config: Any) -> Any:
        """用 LoRA config 包装 base model, 返回 PeftModel (借鉴 PEFT get_peft_model)。"""
        ...

    @abstractmethod
    def save(self, path: str, adapter_name: str = "default") -> None:
        """保存 adapter 权重到目录 (借鉴 PEFT save_pretrained)。"""
        ...

    @abstractmethod
    def load(self, path: str, adapter_name: str = "default") -> None:
        """从目录加载 adapter (借鉴 PEFT PeftModel.load_adapter)。"""
        ...

    @abstractmethod
    def set_active(self, adapter_name: str) -> None:
        """切换激活的 adapter (借鉴 PEFT PeftModel.set_adapter)。"""
        ...

    @abstractmethod
    def list_adapters(self) -> list[str]:
        """列出已加载的 adapter 名。"""
        ...

    @abstractmethod
    def merge_and_unload(self) -> Any:
        """合并 LoRA 权重到 base model, 返回标准 PyTorch model (借鉴 PEFT merge_and_unload)。"""
        ...

    @abstractmethod
    def dump(self, path: str) -> None:
        """pickle 序列化 memory 状态 (借鉴 MemOS LoRAMemory.dump)。"""
        ...

    @abstractmethod
    def load_from(self, path: str) -> None:
        """从 pickle 反序列化 memory 状态 (借鉴 MemOS LoRAMemory.load)。"""
        ...
