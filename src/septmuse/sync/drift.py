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
"""漂移检测 — 形态间数据不一致检测 (架构文档 §4.1 自研)。

检测同一记忆在图/文件/向量三种形态间是否漂移:
- count drift: 各形态记录数不一致
- content drift: 同一 ID 在不同形态中内容不同
- missing drift: 某 ID 在某形态中缺失

权威源: 图 > 文件 > 向量 (架构文档 §4.1)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from septmuse.core.logging import get_logger
from septmuse.sync.synchronizer import AUTHORITY_PRIORITY, SyncForm

logger = get_logger(__name__)


@dataclass
class DriftReport:
    """漂移报告。"""

    memory_id: str
    drifted_forms: list[SyncForm] = field(default_factory=list)
    authority_form: SyncForm | None = None
    details: str = ""

    @property
    def has_drift(self) -> bool:
        return len(self.drifted_forms) > 0


@dataclass
class ConsistencyReport:
    """一致性报告 (全量漂移检测)。"""

    total_checked: int = 0
    drift_count: int = 0
    drifts: list[DriftReport] = field(default_factory=list)
    form_counts: dict[str, int] = field(default_factory=dict)

    @property
    def consistency_rate(self) -> float:
        """一致率 (0-1)。"""
        if self.total_checked == 0:
            return 1.0
        return 1.0 - (self.drift_count / self.total_checked)


class DriftDetector:
    """漂移检测器 (架构文档 §4.1 自研)。

    用法:
        detector = DriftDetector()
        report = detector.check_memory(
            memory_id="mem-123",
            form_contents={
                SyncForm.GRAPH: "alice likes python",
                SyncForm.FILE: "alice likes python",
                SyncForm.VECTOR: "alice likes pythn",  # typo!
            }
        )
        if report.has_drift:
            print(f"Drift detected: {report.drifted_forms}")
    """

    def check_memory(
        self,
        memory_id: str,
        form_contents: dict[SyncForm, str | None],
    ) -> DriftReport:
        """检测单条记忆的形态间漂移。

        Args:
            memory_id: 记忆 ID
            form_contents: 各形态的内容 (None=该形态缺失)
        """
        report = DriftReport(memory_id=memory_id)

        # 找权威源
        authority_content: str | None = None
        for form in AUTHORITY_PRIORITY:
            content = form_contents.get(form)
            if content is not None:
                report.authority_form = form
                authority_content = content
                break

        if authority_content is None:
            report.details = "no content in any form"
            report.drifted_forms = list(form_contents.keys())
            return report

        # 比较各形态与权威源
        for form, content in form_contents.items():
            if form == report.authority_form:
                continue
            if content is None or content != authority_content:
                report.drifted_forms.append(form)

        if report.has_drift:
            authority_name = report.authority_form.value if report.authority_form else "none"
            report.details = f"forms differ from authority ({authority_name})"
            logger.info("drift_detected", memory_id=memory_id, drifted=[f.value for f in report.drifted_forms])

        return report

    def check_consistency(
        self,
        all_form_contents: dict[str, dict[SyncForm, str | None]],
    ) -> ConsistencyReport:
        """全量一致性检测。

        Args:
            all_form_contents: {memory_id: {SyncForm: content}}
        """
        report = ConsistencyReport()
        report.total_checked = len(all_form_contents)

        form_counts: dict[str, int] = {f.value: 0 for f in SyncForm}
        for mid, form_contents in all_form_contents.items():
            drift = self.check_memory(mid, form_contents)
            if drift.has_drift:
                report.drift_count += 1
                report.drifts.append(drift)
            for form, content in form_contents.items():
                if content is not None:
                    form_counts[form.value] += 1

        report.form_counts = form_counts
        logger.info(
            "consistency_check_done",
            total=report.total_checked,
            drifts=report.drift_count,
            rate=report.consistency_rate,
        )
        return report
