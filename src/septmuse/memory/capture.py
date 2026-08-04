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
"""V2 捕获子组件 — 薄包装, 复用 capture/pipeline.py 的 CapturePipeline。

V2Memory 通过 memory/capture.py 引用, 不直接 import capture/。
详见 docs/specs/2026-08-04-v2-memory-architecture.md §4 + §6。
"""

from __future__ import annotations

from septmuse.capture.pipeline import CapturePipeline

__all__ = ["CapturePipeline"]
