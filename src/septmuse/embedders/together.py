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
"""Together AI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

Together API 与 OpenAI Embeddings 兼容, 仅 base_url 和默认 model/dims 不同。
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_DIMS = 1024
BASE_URL = "https://api.together.xyz/v1"


class TogetherEmbedder(_OpenAICompatibleEmbedder):
    """Together AI Embeddings provider (OpenAI 兼容)。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        resolved_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not resolved_key:
            raise ValueError("Together API key required: set TOGETHER_API_KEY or pass api_key=")

        logger.info("embedder_loading", provider="together", model=model, dim=dim)
        client = OpenAI(api_key=resolved_key, base_url=BASE_URL)
        logger.info("embedder_ready", provider="together", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
        self.backend_name = "together"
