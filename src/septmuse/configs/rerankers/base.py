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
"""重排器配置基类。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BaseRerankerConfig(BaseModel):
    """重排器通用配置。"""

    backend: str = Field(default="noop")
    top_k: int | None = Field(default=None, description="重排后返回数量")
