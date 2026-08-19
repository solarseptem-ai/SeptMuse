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
"""共享工具函数 — 消除 storage 层多处重复。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    """当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def is_expired(metadata: dict[str, Any] | None) -> bool:
    """检查 metadata 是否已过期 (基于 expiration_date 字段)。

    统一入口: memory/main.py、pipelines/search.py、memory/async_main.py 共用。
    """
    if not metadata or not isinstance(metadata, dict):
        return False
    exp = metadata.get("expiration_date")
    if not exp or not isinstance(exp, str):
        return False
    try:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    except ValueError:
        return False
    return exp_date < date.today()


def normalize_expiration_date(expiration_date: Any) -> str | None:
    """归一化 expiration_date 为 YYYY-MM-DD 字符串。

    统一入口: memory/main.py、pipelines/add.py、memory/async_main.py 共用。
    """
    if expiration_date is None:
        return None
    if not isinstance(expiration_date, str):
        raise ValueError("expiration_date must be a string in YYYY-MM-DD format")
    try:
        datetime.strptime(expiration_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"expiration_date must be in YYYY-MM-DD format, got: {expiration_date}") from None
    return expiration_date


def rrf_fuse(
    vec_results: list[dict],
    kw_results: list[dict],
    *,
    alpha: float = 0.5,
    k: int = 60,
) -> list[dict]:
    """RRF 融合排序 (k=60 标准参数)。

    score = alpha * 1/(k+rank_vec) + (1-alpha) * 1/(k+rank_kw)
    alpha=1.0 时仅保留向量侧; alpha=0.0 时仅保留关键词侧 (纯模式不混入零分项)。
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if alpha > 0:
        for rank, r in enumerate(vec_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + alpha / (k + rank + 1)
            meta.setdefault(r["id"], r)
    if alpha < 1:
        for rank, r in enumerate(kw_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + (1 - alpha) / (k + rank + 1)
            meta.setdefault(r["id"], r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[mid], "score": sc} for mid, sc in ordered]
