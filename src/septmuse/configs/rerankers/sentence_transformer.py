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
"""SentenceTransformer 重排器配置 (借鉴 mem0 SentenceTransformerRerankerConfig)。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.rerankers.base import BaseRerankerConfig


class SentenceTransformerRerankerConfig(BaseRerankerConfig):
    """sentence-transformers CrossEncoder 配置。"""

    backend: str = Field(default="sentence_transformer")
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    device: str = Field(default="", description="空串=自动检测 cuda/cpu")
    batch_size: int = Field(default=32)
    show_progress_bar: bool = Field(default=False)
    normalize: bool = Field(default=True, description="sigmoid 归一化到 [0,1]")
