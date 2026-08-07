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
"""AWS Bedrock 嵌入 provider — boto3 bedrock-runtime。

按 provider (cohere/titan) 构造不同 body, L2 归一化输出。
"""

from __future__ import annotations

import json
import os

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "amazon.titan-embed-text-v1"


class AWSBedrockEmbedder(Embedder):
    """AWS Bedrock Embeddings provider (titan/cohere, L2 归一化)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_region: str | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 required: pip install septmuse[aws-bedrock]") from e

        self.backend_name = "aws_bedrock"
        self.model = model
        self._dim = embedding_dims
        self._provider = model.split(".")[0]

        access_key = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
        region = aws_region or os.environ.get("AWS_REGION") or "us-west-2"

        logger.info("embedder_loading", provider="aws_bedrock", model=model, region=region)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key if access_key else None,
            aws_secret_access_key=secret_key if secret_key else None,
            aws_session_token=session_token if session_token else None,
        )
        logger.info("embedder_ready", provider="aws_bedrock", model=model)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("AWS Bedrock dimension unknown until first embed() call")
        return self._dim

    def _get_embedding(self, text: str) -> list[float]:
        input_body: dict = {}
        if self._provider == "cohere":
            input_body["input_type"] = "search_document"
            input_body["texts"] = [text]
        else:
            input_body["inputText"] = text
            if self._dim is not None and "v2" in self.model:
                input_body["dimensions"] = self._dim

        body = json.dumps(input_body)
        response = self._client.invoke_model(
            body=body, modelId=self.model, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())

        if self._provider == "cohere":
            return response_body.get("embeddings")[0]
        return response_body.get("embedding")

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        emb = self._get_embedding(text)
        if self._dim is None:
            self._dim = len(emb)
        vec = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]
