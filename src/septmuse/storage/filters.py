"""mem0 风格 filters dict 解析器 — filters dict → SQL WHERE 子句 + 参数。

支持:
- 直接值: {"key": "value"} → key = ?
- 操作符: eq/ne/gt/gte/lt/lte/in/nin/contains/icontains
- 通配符: {"key": "*"} → key IS NOT NULL
- 逻辑运算: AND/OR/NOT
- 实体字段: user_id/agent_id/session_id/run_id/state
- metadata 字段: 任意 key → json_extract(metadata, '$.key')

run_id 映射到 session_id（mem0 兼容）。
"""
from __future__ import annotations

from typing import Any

# memories 表的列名（直接引用，不走 json_extract）
_ENTITY_KEYS = {"user_id", "agent_id", "session_id", "run_id", "state"}

# 逻辑运算符
_LOGICAL_OPS = {"AND", "OR", "NOT"}

# 比较操作符 → SQL 操作符
_COMPARE_OPS = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


class FiltersParser:
    """解析 mem0 风格 filters dict → SQL WHERE 子句 + 参数列表。"""

    def parse(self, filters: dict[str, Any] | None, backend: str = "sqlite") -> tuple[str, list[Any]]:
        """解析 filters → (where_clause, params)。

        Args:
            filters: mem0 风格 filters dict
            backend: "sqlite" 或 "postgres"

        Returns:
            (where_clause, params): WHERE 子句和参数列表。空 filters 返回 ("", [])。
        """
        if not filters:
            return "", []
        return self._parse_dict(filters, backend)

    def _parse_dict(self, filters: dict[str, Any], backend: str) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if key in _LOGICAL_OPS:
                clause, p = self._parse_logical(key, value, backend)
                parts.append(clause)
                params.extend(p)
            elif key in _ENTITY_KEYS:
                actual_key = "session_id" if key == "run_id" else key
                clause, p = self._parse_field(actual_key, value, backend, is_entity=True)
                parts.append(clause)
                params.extend(p)
            else:
                clause, p = self._parse_field(key, value, backend, is_entity=False)
                parts.append(clause)
                params.extend(p)
        return " AND ".join(parts), params

    def _parse_field(self, key: str, value: Any, backend: str, is_entity: bool) -> tuple[str, list[Any]]:
        """解析单个字段条件。"""
        col = key if is_entity else self._metadata_col(key, backend)

        # 通配符 * → IS NOT NULL
        if value == "*":
            return f"{col} IS NOT NULL", []

        # 操作符 dict
        if isinstance(value, dict):
            return self._parse_operator(col, value)

        # 直接值 → 精确匹配
        return f"{col} = ?", [value]

    def _metadata_col(self, key: str, backend: str) -> str:
        """生成 metadata 列引用。"""
        if backend == "postgres":
            return f"metadata->>'{key}'"
        return f"json_extract(metadata, '$.{key}')"

    def _parse_operator(self, col: str, value: dict[str, Any]) -> tuple[str, list[Any]]:
        """解析操作符 dict。"""
        if len(value) != 1:
            raise ValueError(f"操作符 dict 只能有一个 key, got: {list(value.keys())}")

        op = next(iter(value.keys()))
        val = value[op]

        if op in _COMPARE_OPS:
            return f"{col} {_COMPARE_OPS[op]} ?", [val]
        elif op == "in":
            if not isinstance(val, list) or not val:
                raise ValueError(f"in 操作符需要非空列表, got: {val}")
            placeholders = ", ".join("?" * len(val))
            return f"{col} IN ({placeholders})", list(val)
        elif op == "nin":
            if not isinstance(val, list) or not val:
                raise ValueError(f"nin 操作符需要非空列表, got: {val}")
            placeholders = ", ".join("?" * len(val))
            return f"{col} NOT IN ({placeholders})", list(val)
        elif op == "contains":
            return f"{col} LIKE '%' || ? || '%'", [val]
        elif op == "icontains":
            return f"LOWER({col}) LIKE LOWER('%' || ? || '%')", [val]
        else:
            raise ValueError(f"不支持的操作符: {op}")

    def _parse_logical(self, op: str, conditions: list[dict], backend: str) -> tuple[str, list[Any]]:
        """解析逻辑运算（AND/OR/NOT）。"""
        if not isinstance(conditions, list):
            raise ValueError(f"{op} 需要列表值, got: {type(conditions)}")

        parts: list[str] = []
        params: list[Any] = []
        for cond in conditions:
            clause, p = self._parse_dict(cond, backend)
            parts.append(f"({clause})")
            params.extend(p)

        if op == "NOT":
            inner = " AND ".join(parts)
            return f"NOT ({inner})", params
        else:
            joined = f" {op} ".join(parts)
            return f"({joined})", params
