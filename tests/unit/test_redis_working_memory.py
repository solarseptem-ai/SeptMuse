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
"""RedisWorkingMemoryStore 单元测试。

单元测试: mock Redis client, 不需要真实 Redis。
集成测试: @pytest.mark.skipif 需要 SEPTMUSE_TEST_REDIS_URL 环境变量 + 真实 Redis。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from septmuse.models.block import Block
from septmuse.storage.working_memory_stores.base import WorkingMemoryStore
from septmuse.storage.working_memory_stores.redis_store import (
    RedisWorkingMemoryStore,
    _block_to_dict,
    _dict_to_block,
)

# 集成测试 gate: 需要环境变量 SEPTMUSE_TEST_REDIS_URL 指向真实 Redis
HAS_REDIS_URL = bool(os.getenv("SEPTMUSE_TEST_REDIS_URL"))
REDIS_URL = os.getenv("SEPTMUSE_TEST_REDIS_URL", "redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# 序列化辅助函数测试 (不需要 Redis)
# ---------------------------------------------------------------------------


class TestSerialization:
    """_block_to_dict / _dict_to_block 往返测试。"""

    def test_roundtrip_all_fields(self):
        """Block → dict → Block 保持所有字段。"""
        now = datetime.now(timezone.utc)
        block = Block(
            id="block-test-1",
            agent_id="agent-1",
            label="human",
            value="喜欢 Python",
            limit=3000,
            read_only=True,
            description="用户画像",
            tags=["profile", "preference"],
            created_at=now,
            updated_at=now,
        )
        d = _block_to_dict(block)
        restored = _dict_to_block(d)

        assert restored.id == block.id
        assert restored.agent_id == block.agent_id
        assert restored.label == block.label
        assert restored.value == block.value
        assert restored.limit == block.limit
        assert restored.read_only is True
        assert restored.description == block.description
        assert list(restored.tags) == ["profile", "preference"]
        assert restored.created_at == block.created_at
        assert restored.updated_at == block.updated_at

    def test_roundtrip_empty_tags(self):
        """空 tags 列表正确往返。"""
        block = Block(agent_id="a1", label="persona", value="")
        d = _block_to_dict(block)
        assert d["tags"] == []
        restored = _dict_to_block(d)
        assert list(restored.tags) == []

    def test_roundtrip_none_description(self):
        """description=None 正确处理。"""
        block = Block(agent_id="a1", label="task", value="do something")
        d = _block_to_dict(block)
        assert d["description"] is None
        restored = _dict_to_block(d)
        assert restored.description is None

    def test_datetime_isoformat(self):
        """datetime 字段序列化为 ISO 字符串。"""
        ts = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        block = Block(agent_id="a", label="l", value="v", created_at=ts, updated_at=ts)
        d = _block_to_dict(block)
        assert isinstance(d["created_at"], str)
        assert d["created_at"] == ts.isoformat()
        restored = _dict_to_block(d)
        assert restored.created_at == ts

    def test_roundtrip_unicode(self):
        """Unicode 文本正确往返。"""
        block = Block(agent_id="a", label="l", value="喜欢 Python \x1f emoji")
        d = _block_to_dict(block)
        restored = _dict_to_block(d)
        assert restored.value == block.value


# ---------------------------------------------------------------------------
# Mock 单元测试 (mock Redis client, 不需要真实 Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """创建 RedisWorkingMemoryStore, mock 内部 Redis client。"""
    store = RedisWorkingMemoryStore(url=REDIS_URL)
    store._redis = MagicMock()
    return store


class TestRedisStoreContract:
    """ABC 契约测试。"""

    def test_is_working_memory_store(self, mock_store):
        """RedisWorkingMemoryStore 是 WorkingMemoryStore 子类。"""
        assert isinstance(mock_store, WorkingMemoryStore)

    def test_implements_all_abstract_methods(self, mock_store):
        """所有抽象方法都已实现 (能实例化即证明)。"""
        for method in ["get_blocks", "save_block", "update_block_value", "delete_block", "ensure_default_blocks"]:
            assert callable(getattr(mock_store, method))


class TestRedisStoreKey:
    """Redis key 构造测试。"""

    def test_key_format(self):
        """key 格式为 septmuse:wm:{agent_id}。"""
        assert RedisWorkingMemoryStore._key("agent-1") == "septmuse:wm:agent-1"
        assert RedisWorkingMemoryStore._key("alice") == "septmuse:wm:alice"


class TestGetBlocks:
    """get_blocks 测试。"""

    def test_empty(self, mock_store):
        """无数据时返回空列表。"""
        mock_store._redis.hgetall.return_value = {}
        result = mock_store.get_blocks("agent-1")
        assert result == []

    def test_returns_blocks(self, mock_store):
        """有数据时返回 Block 列表。"""
        block = Block(agent_id="agent-1", label="human", value="hello")
        d = _block_to_dict(block)
        mock_store._redis.hgetall.return_value = {"human": json.dumps(d)}
        result = mock_store.get_blocks("agent-1")
        assert len(result) == 1
        assert result[0].label == "human"
        assert result[0].value == "hello"
        assert result[0].agent_id == "agent-1"

    def test_multiple_blocks(self, mock_store):
        """多个 block 都正确返回。"""
        b1 = Block(agent_id="a1", label="human", value="v1")
        b2 = Block(agent_id="a1", label="persona", value="v2")
        mock_store._redis.hgetall.return_value = {
            "human": json.dumps(_block_to_dict(b1)),
            "persona": json.dumps(_block_to_dict(b2)),
        }
        result = mock_store.get_blocks("a1")
        labels = {b.label for b in result}
        assert labels == {"human", "persona"}

    def test_calls_hgetall_with_correct_key(self, mock_store):
        """使用正确的 key 调用 hgetall。"""
        mock_store._redis.hgetall.return_value = {}
        mock_store.get_blocks("agent-42")
        mock_store._redis.hgetall.assert_called_once_with("septmuse:wm:agent-42")


class TestSaveBlock:
    """save_block 测试。"""

    def test_calls_hset(self, mock_store):
        """正确调用 HSET。"""
        block = Block(agent_id="agent-1", label="human", value="hello")
        mock_store.save_block(block)

        mock_store._redis.hset.assert_called_once()
        call_args = mock_store._redis.hset.call_args
        assert call_args.args[0] == "septmuse:wm:agent-1"
        assert call_args.args[1] == "human"

    def test_serializes_to_json(self, mock_store):
        """value 是 JSON 字符串。"""
        block = Block(agent_id="agent-1", label="human", value="hello", tags=["a", "b"])
        mock_store.save_block(block)

        call_args = mock_store._redis.hset.call_args
        stored = json.loads(call_args.args[2])
        assert stored["value"] == "hello"
        assert stored["agent_id"] == "agent-1"
        assert stored["tags"] == ["a", "b"]

    def test_touches_updated_at(self, mock_store):
        """保存时更新 updated_at。"""
        old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        block = Block(agent_id="a", label="l", value="v", updated_at=old_ts)
        mock_store.save_block(block)
        assert block.updated_at > old_ts

    def test_returns_block(self, mock_store):
        """返回保存的 block。"""
        block = Block(agent_id="a", label="l", value="v")
        result = mock_store.save_block(block)
        assert result is block


class TestUpdateBlockValue:
    """update_block_value 测试。"""

    def test_updates_value(self, mock_store):
        """找到 block 时更新 value。"""
        block = Block(agent_id="a1", label="human", value="old")
        mock_store._redis.hget.return_value = json.dumps(_block_to_dict(block))

        result = mock_store.update_block_value("a1", "human", "new value")
        assert result is not None
        assert result.value == "new value"
        assert result.label == "human"
        assert result.agent_id == "a1"

    def test_persists_update(self, mock_store):
        """更新后写回 Redis。"""
        block = Block(agent_id="a1", label="human", value="old")
        mock_store._redis.hget.return_value = json.dumps(_block_to_dict(block))

        mock_store.update_block_value("a1", "human", "new")
        mock_store._redis.hset.assert_called_once()

    def test_not_found_returns_none(self, mock_store):
        """label 不存在时返回 None。"""
        mock_store._redis.hget.return_value = None
        result = mock_store.update_block_value("a1", "nonexistent", "val")
        assert result is None

    def test_calls_hget_with_correct_key(self, mock_store):
        """使用正确的 key + label 调用 hget。"""
        mock_store._redis.hget.return_value = None
        mock_store.update_block_value("agent-x", "persona", "val")
        mock_store._redis.hget.assert_called_once_with("septmuse:wm:agent-x", "persona")


class TestDeleteBlock:
    """delete_block 测试。"""

    def test_deletes_found(self, mock_store):
        """找到并删除。"""
        mock_store._redis.hdel.return_value = 1
        result = mock_store.delete_block("a1", "human")
        assert result is True

    def test_not_found_returns_false(self, mock_store):
        """label 不存在时返回 False。"""
        mock_store._redis.hdel.return_value = 0
        result = mock_store.delete_block("a1", "nonexistent")
        assert result is False

    def test_calls_hdel_with_correct_key(self, mock_store):
        """使用正确的 key + label 调用 hdel。"""
        mock_store._redis.hdel.return_value = 1
        mock_store.delete_block("agent-99", "task")
        mock_store._redis.hdel.assert_called_once_with("septmuse:wm:agent-99", "task")


class TestEnsureDefaultBlocks:
    """ensure_default_blocks 测试。"""

    def test_creates_when_empty(self, mock_store):
        """无 block 时创建默认 human + persona。"""
        # 第一次 get_blocks 返回空 → 触发创建
        # 第二次 get_blocks 返回创建后的 block
        mock_store._redis.hgetall.return_value = {}

        # save_block 会调用 hset, 但我们 mock 了 _redis 所以不会真正写
        # 但第二次 get_blocks 仍然返回空 (因为 mock)
        # ensure_default_blocks 的实现: 先 get_blocks, 空则 save * N, 再 get_blocks
        # 由于 mock 的 hgetall 永远返回 {}, 第二次 get_blocks 也返回 []
        # 这是 mock 的局限性, 我们只验证 save_block 被调用
        mock_store.ensure_default_blocks("agent-1")

        # 应该调用了 2 次 hset (human + persona)
        assert mock_store._redis.hset.call_count == 2

    def test_skips_when_not_empty(self, mock_store):
        """有 block 时不创建。"""
        block = Block(agent_id="a1", label="human", value="existing")
        mock_store._redis.hgetall.return_value = {"human": json.dumps(_block_to_dict(block))}

        result = mock_store.ensure_default_blocks("a1")

        # 不应该调用 hset
        mock_store._redis.hset.assert_not_called()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 集成测试 (需要真实 Redis, skipif)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not HAS_REDIS_URL, reason="需要 SEPTMUSE_TEST_REDIS_URL 环境变量")
class TestRedisIntegration:
    """Redis 集成测试 — CRUD 全流程 (需要真实 Redis)。"""

    def test_full_crud_lifecycle(self):
        """完整 CRUD 生命周期。"""
        store = RedisWorkingMemoryStore(url=REDIS_URL)
        agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"

        try:
            # 1. 初始为空
            assert store.get_blocks(agent_id) == []

            # 2. ensure_default_blocks 创建 human + persona
            blocks = store.ensure_default_blocks(agent_id)
            assert len(blocks) == 2
            labels = {b.label for b in blocks}
            assert labels == {"human", "persona"}

            # 3. update_block_value
            updated = store.update_block_value(agent_id, "human", "喜欢 Python")
            assert updated is not None
            assert updated.value == "喜欢 Python"

            # 4. get_blocks 反映更新
            blocks = store.get_blocks(agent_id)
            human = next(b for b in blocks if b.label == "human")
            assert human.value == "喜欢 Python"

            # 5. save_block (upsert)
            new_block = Block(agent_id=agent_id, label="task", value="写测试", limit=1500)
            store.save_block(new_block)
            blocks = store.get_blocks(agent_id)
            assert len(blocks) == 3

            # 6. delete_block
            assert store.delete_block(agent_id, "task") is True
            assert store.delete_block(agent_id, "task") is False  # 已删除

            # 7. update_block_value 不存在的 label
            assert store.update_block_value(agent_id, "nonexistent", "val") is None

        finally:
            # 清理: 删除测试 agent 的所有 block
            store._redis.delete(f"septmuse:wm:{agent_id}")

    def test_default_blocks_idempotent(self):
        """ensure_default_blocks 幂等 (重复调用不创建重复 block)。"""
        store = RedisWorkingMemoryStore(url=REDIS_URL)
        agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"

        try:
            b1 = store.ensure_default_blocks(agent_id)
            b2 = store.ensure_default_blocks(agent_id)
            assert len(b1) == len(b2) == 2
        finally:
            store._redis.delete(f"septmuse:wm:{agent_id}")
