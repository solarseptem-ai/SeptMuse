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
"""spaCy 实体抽取配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.extraction.base import BaseEntityExtractorConfig


class SpacyExtractorConfig(BaseEntityExtractorConfig):
    """spaCy NER 配置。"""

    backend: str = Field(default="spacy")
    model_name: str = Field(default="en_core_web_sm", description="spaCy 模型名")
