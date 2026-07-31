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
"""src.septmuse.storage.parametric package — LoRA 参数化记忆 (借鉴 HuggingFace PEFT)。

仅当使用自托管模型 (HuggingFace 后端) 时启用; API/闭源模型跳过此格。
依赖反转: 不实现训练, 仅做 adapter 生命周期管理 (wrap/save/load/set_active/merge_and_unload)。
"""

from septmuse.storage.parametric.base import BaseParametricMemory
from septmuse.storage.parametric.lora import LoRAConfig, LoRAMemory

__all__ = ["BaseParametricMemory", "LoRAConfig", "LoRAMemory"]
