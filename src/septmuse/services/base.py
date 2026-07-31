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
"""存储服务基类 (借鉴 Langflow services/base.py Service ABC)。

所有存储服务继承此 ABC, 对外暴露统一接口, 内部委托给 SeptMuse storage/
下的具体后端实现 (CAMEL 风格的正交 ABC 矩阵)。
"""

from __future__ import annotations

from abc import ABC


class Service(ABC):
    """存储服务抽象基类 (借鉴 Langflow Service ABC)。

    子类需设置 ``name`` 属性, 在 __init__ 末尾调用 set_ready()。
    """

    name: str
    ready: bool = False

    def get_schema(self) -> dict:
        """反射列出所有公开方法及其参数 / 返回类型 / 文档 (Langflow 模式)。"""
        schema: dict[str, dict] = {}
        ignore = ["teardown", "set_ready"]
        for method in dir(self):
            if method.startswith("_") or method in ignore:
                continue
            func = getattr(self, method)
            if not callable(func):
                continue
            schema[method] = {
                "name": method,
                "parameters": getattr(func, "__annotations__", {}),
                "return": getattr(func, "__annotations__", {}).get("return"),
                "documentation": func.__doc__,
            }
        return schema

    def set_ready(self) -> None:
        self.ready = True

    async def teardown(self) -> None:
        """释放底层资源 (子类可覆盖, 委托后端 close)。"""
        return
