"""models/ 包测试 — 验证表定义可建表、列完整。"""

# 表类的 import 仅用于副作用: 触发 SQLModel.metadata 注册表定义, 测试体用字符串表名引用。
# ruff: noqa: F401

import pytest
from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from septmuse.services.database.models import (
    AccessLogTable,
    EntityRelationTable,
    EntityTable,
    HistoryTable,
    MemoryTable,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


def test_memory_table_columns(engine):
    """MemoryTable 有全部 16 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("memories")}
    expected = {
        "id", "user_id", "agent_id", "session_id", "content", "embedding",
        "metadata", "created_at", "updated_at", "is_deleted", "state",
        "app_id", "archived_at", "deleted_at", "valid_at", "invalid_at", "expired_at",
    }
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_history_table_columns(engine):
    """HistoryTable 有全部 7 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("history")}
    expected = {"id", "memory_id", "old_memory", "new_memory", "event", "created_at", "is_deleted"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_access_log_table_columns(engine):
    """AccessLogTable 有全部 6 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("memory_access_logs")}
    expected = {"id", "memory_id", "app_id", "access_type", "metadata", "accessed_at"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_entity_table_columns(engine):
    """EntityTable 有全部 10 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("septmuse_entities")}
    expected = {
        "id", "entity_text", "entity_type", "entity_embedding",
        "linked_memory_ids", "user_id", "agent_id", "created_at", "updated_at", "is_deleted",
    }
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_entity_relation_table_columns(engine):
    """EntityRelationTable 有全部 7 列（含 memory_id）。"""
    cols = {c["name"] for c in inspect(engine).get_columns("entity_relations")}
    expected = {"id", "source_entity", "relation", "target_entity", "user_id", "memory_id", "created_at"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_all_tables_registered():
    """5 个表类都注册到 SQLModel.metadata。"""
    table_names = set(SQLModel.metadata.tables.keys())
    assert "memories" in table_names
    assert "history" in table_names
    assert "memory_access_logs" in table_names
    assert "septmuse_entities" in table_names
    assert "entity_relations" in table_names
