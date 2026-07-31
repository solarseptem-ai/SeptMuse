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
"""Cross-encoder 重排器配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.rerankers.base import BaseRerankerConfig


class CrossEncoderRerankerConfig(BaseRerankerConfig):
    """ONNX cross-encoder 配置。"""

    backend: str = Field(default="cross_encoder")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    model_cache_dir: str = Field(default="")
