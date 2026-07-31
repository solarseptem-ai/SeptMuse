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
"""ONNX 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class OnnxEmbedderConfig(BaseEmbedderConfig):
    """ONNX 嵌入配置 (无 torch, ModelScope 下载)。"""

    backend: str = Field(default="onnx")
    model: str = Field(default="Xenova/all-MiniLM-L6-v2")
    embedding_dims: int = Field(default=384)
    model_cache_dir: str = Field(default="", description="空字符串 → ~/.septmuse/models/")
