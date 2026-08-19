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
"""输入校验工具。"""

from __future__ import annotations


def validate_entity_id(value: str | None, name: str = "entity_id") -> str | None:
    """校验并 trim entity ID。"""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"Invalid {name}: cannot be empty or whitespace-only.")
    if any(c.isspace() for c in trimmed):
        raise ValueError(f"Invalid {name}: cannot contain internal whitespace.")
    return trimmed


def validate_search_params(threshold: float | None = None, top_k: int | None = None) -> None:
    """校验 search 参数。"""
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(f"threshold must be a number, got {type(threshold).__name__}")
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k < 0:
            raise ValueError(f"Invalid top_k: {top_k}. Must be non-negative.")


def validate_search_query(query: str) -> str:
    """校验并 trim search query。"""
    if not isinstance(query, str):
        raise ValueError(f"Invalid query: must be a string, got {type(query).__name__}.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed
