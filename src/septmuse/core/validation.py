"""输入校验工具 (对齐 mem0 _validate_and_trim_entity_id 等)."""

from __future__ import annotations


def validate_entity_id(value: str | None, name: str = "entity_id") -> str | None:
    """校验并 trim entity ID (user_id/agent_id/session_id).

    - None → None (可选参数未提供)
    - 非字符串 → str(value)
    - trim 前后空格
    - 拒空/纯空格 → ValueError
    - 拒内部空格 → ValueError

    Returns: trimmed entity_id
    """
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
    """校验 search 参数.

    - threshold: [0, 1] 范围
    - top_k: 非负整数

    Raises: ValueError if invalid
    """
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
    """校验并 trim search query.

    - 必须是字符串
    - trim 前后空格
    - 拒空/纯空格 → ValueError

    Returns: trimmed query
    """
    if not isinstance(query, str):
        raise ValueError(f"Invalid query: must be a string, got {type(query).__name__}.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed
