"""FiltersParser 测试 — mem0 风格 filters dict → SQL WHERE 子句。"""
import pytest

from septmuse.storage.filters import FiltersParser


@pytest.fixture
def parser():
    return FiltersParser()


def test_empty_filters(parser):
    """空 filters → 空 WHERE 子句。"""
    clause, params = parser.parse({})
    assert clause == ""
    assert params == []


def test_none_filters(parser):
    """None filters → 空 WHERE 子句。"""
    clause, params = parser.parse(None)
    assert clause == ""
    assert params == []


def test_direct_value_exact_match(parser):
    """直接值 → 精确匹配。"""
    clause, params = parser.parse({"user_id": "alice"})
    assert "user_id = ?" in clause
    assert params == ["alice"]


def test_eq_operator(parser):
    """eq 操作符。"""
    clause, params = parser.parse({"category": {"eq": "work"}})
    assert "json_extract(metadata, '$.category') = ?" in clause
    assert params == ["work"]


def test_ne_operator(parser):
    """ne 操作符。"""
    clause, params = parser.parse({"category": {"ne": "work"}})
    assert "json_extract(metadata, '$.category') <> ?" in clause
    assert params == ["work"]


def test_gt_gte_lt_lte_operators(parser):
    """数值比较操作符。"""
    for op, sql_op in [("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")]:
        clause, params = parser.parse({"priority": {op: 5}})
        assert f"json_extract(metadata, '$.priority') {sql_op} ?" in clause
        assert params == [5]


def test_in_operator(parser):
    """in 操作符。"""
    clause, params = parser.parse({"tag": {"in": ["urgent", "bug"]}})
    assert "IN (?, ?)" in clause
    assert params == ["urgent", "bug"]


def test_nin_operator(parser):
    """nin 操作符。"""
    clause, params = parser.parse({"tag": {"nin": ["archived"]}})
    assert "NOT IN (?)" in clause
    assert params == ["archived"]


def test_contains_operator(parser):
    """contains 操作符。"""
    clause, params = parser.parse({"content": {"contains": "Python"}})
    assert "LIKE" in clause
    assert params == ["Python"]


def test_icontains_operator(parser):
    """icontains 操作符（不区分大小写）。"""
    clause, params = parser.parse({"content": {"icontains": "python"}})
    assert "LOWER" in clause
    assert params == ["python"]


def test_wildcard(parser):
    """通配符 * → IS NOT NULL。"""
    clause, params = parser.parse({"category": "*"})
    assert "IS NOT NULL" in clause
    assert params == []


def test_and_logical(parser):
    """AND 逻辑运算。"""
    clause, params = parser.parse({"AND": [{"user_id": "alice"}, {"session_id": "s1"}]})
    assert "AND" in clause
    assert "user_id = ?" in clause
    assert "session_id = ?" in clause
    assert params == ["alice", "s1"]


def test_or_logical(parser):
    """OR 逻辑运算。"""
    clause, params = parser.parse({"OR": [{"agent_id": "a1"}, {"agent_id": "a2"}]})
    assert "OR" in clause
    assert params == ["a1", "a2"]


def test_not_logical(parser):
    """NOT 逻辑运算。"""
    clause, params = parser.parse({"NOT": [{"state": "deleted"}]})
    assert "NOT" in clause
    assert "state = ?" in clause
    assert params == ["deleted"]


def test_run_id_maps_to_session_id(parser):
    """run_id 映射到 session_id（mem0 兼容）。"""
    clause, params = parser.parse({"run_id": "sess-123"})
    assert "session_id = ?" in clause
    assert params == ["sess-123"]


def test_entity_field_vs_metadata(parser):
    """实体字段直接引用列名，metadata 字段用 json_extract。"""
    clause, _ = parser.parse({"user_id": "alice", "category": "work"})
    assert "user_id = ?" in clause
    assert "json_extract(metadata, '$.category')" in clause


def test_postgres_backend(parser):
    """PG 后端用 metadata->>'key' 替代 json_extract。"""
    clause, _ = parser.parse({"category": "work"}, backend="postgres")
    assert "metadata->>'category'" in clause
    assert "json_extract" not in clause


def test_nested_logical(parser):
    """嵌套逻辑运算：AND(OR(...), NOT(...))。"""
    clause, params = parser.parse({
        "AND": [
            {"OR": [{"user_id": "a"}, {"user_id": "b"}]},
            {"NOT": [{"state": "deleted"}]},
        ]
    })
    assert "OR" in clause
    assert "NOT" in clause
    assert params == ["a", "b", "deleted"]
