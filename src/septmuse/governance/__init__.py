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
"""septmuse.governance — 治理 (access/audit/approval/rbac/sharing)。"""

from septmuse.governance.access import (
    MemoryState,
    async_check_memory_access_permissions,
    check_memory_access_permissions,
)
from septmuse.governance.approval import DedupWindow, WriteValidator
from septmuse.governance.audit import async_record_access, record_access
from septmuse.governance.rbac import AccessCheckResult, AccessGrant, Permission, RBACManager, Role
from septmuse.governance.sharing import MemoryScope, SharedMemoryAccessor

__all__ = [
    "AccessCheckResult",
    "AccessGrant",
    "DedupWindow",
    "MemoryScope",
    "MemoryState",
    "Permission",
    "RBACManager",
    "Role",
    "SharedMemoryAccessor",
    "WriteValidator",
    "async_check_memory_access_permissions",
    "async_record_access",
    "check_memory_access_permissions",
    "record_access",
]
