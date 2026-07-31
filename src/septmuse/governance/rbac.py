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
"""跨 agent 权限治理 — RBAC 角色+权限矩阵 (架构文档 §7.2 自研)。

Agno 仅共享无权限, 需 RBAC。SeptMuse 新增角色+权限矩阵:
- owner: 用户本人, 全权限
- agent: 被 owner 授权的 agent, 读写指定 namespace
- observer: 只读

权限: read / write / admin
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class Role(str, Enum):
    """RBAC 角色。"""

    OWNER = "owner"  # 用户本人, 全权限
    AGENT = "agent"  # 被 owner 授权的 agent
    OBSERVER = "observer"  # 只读


class Permission(str, Enum):
    """权限类型。"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# 权限矩阵: role → allowed permissions
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.OWNER: {Permission.READ, Permission.WRITE, Permission.ADMIN},
    Role.AGENT: {Permission.READ, Permission.WRITE},
    Role.OBSERVER: {Permission.READ},
}


@dataclass
class AccessGrant:
    """权限授予。"""

    user_id: str  # 被授权的用户
    agent_id: str  # 授权的 agent
    role: Role = Role.AGENT
    namespaces: list[str] = field(default_factory=lambda: ["*"])  # * = 全部 namespace


@dataclass
class AccessCheckResult:
    """权限检查结果。"""

    allowed: bool
    role: Role | None = None
    permission: Permission | None = None
    reason: str = ""


class RBACManager:
    """RBAC 权限管理器 (架构文档 §7.2 自研)。

    用法:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT, namespaces=["semantic"])
        result = rbac.check("alice", "bot1", Permission.WRITE, namespace="semantic")
        assert result.allowed
    """

    def __init__(self) -> None:
        self._grants: dict[str, list[AccessGrant]] = {}  # user_id → grants

    def grant(
        self,
        user_id: str,
        agent_id: str,
        role: Role = Role.AGENT,
        namespaces: list[str] | None = None,
    ) -> AccessGrant:
        """授予权限。"""
        g = AccessGrant(
            user_id=user_id,
            agent_id=agent_id,
            role=role,
            namespaces=namespaces or ["*"],
        )
        self._grants.setdefault(user_id, []).append(g)
        logger.info("rbac_grant", user_id=user_id, agent_id=agent_id, role=role.value)
        return g

    def revoke(self, user_id: str, agent_id: str) -> bool:
        """撤销权限。"""
        grants = self._grants.get(user_id, [])
        before = len(grants)
        self._grants[user_id] = [g for g in grants if g.agent_id != agent_id]
        return len(self._grants[user_id]) < before

    def check(
        self,
        user_id: str,
        agent_id: str,
        permission: Permission,
        namespace: str = "*",
    ) -> AccessCheckResult:
        """检查权限。"""
        # owner 对自己有全权限
        if agent_id == user_id or agent_id == "self":
            return AccessCheckResult(allowed=True, role=Role.OWNER, permission=permission, reason="self access")

        grants = self._grants.get(user_id, [])
        for g in grants:
            if g.agent_id != agent_id:
                continue
            # 检查 namespace
            ns_ok = "*" in g.namespaces or namespace in g.namespaces
            if not ns_ok:
                continue
            # 检查权限
            allowed_perms = ROLE_PERMISSIONS.get(g.role, set())
            if permission in allowed_perms:
                return AccessCheckResult(allowed=True, role=g.role, permission=permission, reason="granted")

        return AccessCheckResult(allowed=False, reason="no matching grant")

    def list_grants(self, user_id: str) -> list[AccessGrant]:
        """列出用户的所有授权。"""
        return list(self._grants.get(user_id, []))

    def list_agents_for_user(self, user_id: str) -> list[str]:
        """列出能访问该用户记忆的所有 agent。"""
        return [g.agent_id for g in self._grants.get(user_id, [])]

    def has_any_access(self, user_id: str, agent_id: str) -> bool:
        """检查 agent 是否有任何访问权限。"""
        return any(g.agent_id == agent_id for g in self._grants.get(user_id, []))
