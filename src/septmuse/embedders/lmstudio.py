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
"""LM Studio 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

LM Studio 本地服务, OpenAI 兼容 API。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf"
DEFAULT_DIMS = 1536
DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioEmbedder(_OpenAICompatibleEmbedder):
    """LM Studio Embeddings provider (OpenAI 兼容, 本地)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        lmstudio_base_url: str = DEFAULT_BASE_URL,
        embedding_dims: int | None = None,
        api_key: str = "lm-studio",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        logger.info("embedder_loading", provider="lmstudio", model=model, dim=dim, base_url=lmstudio_base_url)
        client = OpenAI(base_url=lmstudio_base_url, api_key=api_key)
        logger.info("embedder_ready", provider="lmstudio", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
        self.backend_name = "lmstudio"
