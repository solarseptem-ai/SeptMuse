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
"""Azure OpenAI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

支持 API key 或 DefaultAzureCredential AD token provider 认证。
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_DIMS = 1536


class AzureOpenAIEmbedder(_OpenAICompatibleEmbedder):
    """Azure OpenAI Embeddings provider (OpenAI 兼容 + AD token)。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        azure_deployment: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        resolved_key = api_key or os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY")
        resolved_deployment = azure_deployment or os.getenv("EMBEDDING_AZURE_DEPLOYMENT")
        resolved_endpoint = azure_endpoint or os.getenv("EMBEDDING_AZURE_ENDPOINT")
        resolved_api_version = api_version or os.getenv("EMBEDDING_AZURE_API_VERSION")

        azure_ad_token_provider = None
        if not resolved_key or resolved_key in ("", "your-api-key"):
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                credential = DefaultAzureCredential()
                azure_ad_token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                resolved_key = None
            except ImportError:
                pass

        logger.info("embedder_loading", provider="azure_openai", model=model, dim=dim)
        client = AzureOpenAI(
            azure_deployment=resolved_deployment,
            azure_endpoint=resolved_endpoint,
            azure_ad_token_provider=azure_ad_token_provider,
            api_version=resolved_api_version,
            api_key=resolved_key,
        )
        logger.info("embedder_ready", provider="azure_openai", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
        self.backend_name = "azure_openai"
