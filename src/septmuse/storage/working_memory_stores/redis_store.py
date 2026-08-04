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
"""Redis 工作记忆后端 — Block CRUD, Redis hash 存储。

可选后端: pip install septmuse[redis]
配置: SEPTMUSE_WORKING_MEMORY_BACKEND=redis + SEPTMUSE_REDIS_URL=redis://localhost:6379/0

数据结构:
    Key:   septmuse:wm:{agent_id}   (Redis hash)
    Field: block label (如 "human" / "persona")
    Value: JSON 序列化的 Block dict
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.models.block import Block, default_blocks
from septmuse.storage.working_memory_stores.base import WorkingMemoryStore

logger = get_logger(__name__)

_KEY_PREFIX = "septmuse:wm:"


def _block_to_dict(block: Block) -> dict[str, Any]:
    """Block 转 JSON 可序列化 dict。"""
    return {
        "id": block.id,
        "agent_id": block.agent_id,
        "label": block.label,
        "value": block.value,
        "limit": block.limit,
        "read_only": block.read_only,
        "description": block.description,
        "tags": list(block.tags),
        "created_at": block.created_at.isoformat(),
        "updated_at": block.updated_at.isoformat(),
    }


def _dict_to_block(d: dict[str, Any]) -> Block:
    """dict 转 Block。"""
    return Block(
        id=d["id"],
        agent_id=d["agent_id"],
        label=d["label"],
        value=d["value"],
        limit=d["limit"],
        read_only=d["read_only"],
        description=d.get("description"),
        tags=list(d.get("tags", [])),
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )


class RedisWorkingMemoryStore(WorkingMemoryStore):
    """Redis 工作记忆后端 (Redis hash 存储)。

    用法:
        store = RedisWorkingMemoryStore(url="redis://localhost:6379/0")
        blocks = store.ensure_default_blocks("agent-1")
    """

    def __init__(self, url: str) -> None:
        import redis  # 延迟 import, factory fallback 依赖此 ImportError

        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._url = url

    @staticmethod
    def _key(agent_id: str) -> str:
        """构造 Redis key。"""
        return f"{_KEY_PREFIX}{agent_id}"

    def get_blocks(self, agent_id: str) -> list[Block]:
        """加载 agent 的全部 block。"""
        raw = self._redis.hgetall(self._key(agent_id))
        if not raw:
            return []
        return [_dict_to_block(json.loads(v)) for v in raw.values()]

    def save_block(self, block: Block) -> Block:
        """保存 block (upsert, 按 agent_id + label)。"""
        block.touch()
        key = self._key(block.agent_id)
        self._redis.hset(key, block.label, json.dumps(_block_to_dict(block)))
        return block

    def update_block_value(self, agent_id: str, label: str, value: str) -> Block | None:
        """更新 block value。"""
        key = self._key(agent_id)
        raw = self._redis.hget(key, label)
        if not raw:
            return None
        d: dict[str, Any] = json.loads(raw)
        d["value"] = value
        d["updated_at"] = datetime.now(timezone.utc).isoformat()
        block = _dict_to_block(d)
        self._redis.hset(key, label, json.dumps(_block_to_dict(block)))
        return block

    def delete_block(self, agent_id: str, label: str) -> bool:
        """删除 block。"""
        deleted = self._redis.hdel(self._key(agent_id), label)
        return deleted > 0

    def ensure_default_blocks(self, agent_id: str) -> list[Block]:
        """确保 agent 有默认 block (human + persona), 无则创建。"""
        blocks = self.get_blocks(agent_id)
        if blocks:
            return blocks
        for block in default_blocks(agent_id):
            self.save_block(block)
        logger.info("default_blocks_created", agent_id=agent_id, backend="redis")
        return self.get_blocks(agent_id)
