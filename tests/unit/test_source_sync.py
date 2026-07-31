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
"""阶段5 §4.1 源同步器单元测试 — synchronizer + drift。

固化 (架构文档 §4.1 自研):
- SourceSynchronizer: 并行写多形态 + 补偿任务 + 权威源重同步
- DriftDetector: 形态间漂移检测 (count/content/missing drift)
"""

from __future__ import annotations

import pytest

from septmuse.sync.drift import ConsistencyReport, DriftDetector, DriftReport
from septmuse.sync.synchronizer import (
    AUTHORITY_PRIORITY,
    CompensationTask,
    SourceSynchronizer,
    SyncForm,
    SyncResult,
    SyncWriteResult,
)

# ======================================================================
# SourceSynchronizer
# ======================================================================


class TestSourceSynchronizer:
    def test_register_and_write(self) -> None:
        sync = SourceSynchronizer()
        sync.register(SyncForm.VECTOR, lambda content, meta: f"vec-{content[:5]}")
        result = sync.write("hello world", {"user_id": "alice"}, targets=[SyncForm.VECTOR])
        assert result.all_succeeded
        assert len(result.writes) == 1
        assert result.writes[0].success

    def test_parallel_write_multiple_forms(self) -> None:
        sync = SourceSynchronizer()
        sync.register(SyncForm.VECTOR, lambda c, m: f"vec-{c[:3]}")
        sync.register(SyncForm.FILE, lambda c, m: f"file-{c[:3]}")
        sync.register(SyncForm.GRAPH, lambda c, m: f"graph-{c[:3]}")
        result = sync.write("hello", {"user_id": "alice"})
        assert len(result.writes) == 3
        assert result.all_succeeded

    def test_write_failure_creates_compensation(self) -> None:
        sync = SourceSynchronizer()

        def failing_writer(content: str, meta: dict) -> str:
            raise RuntimeError("write failed")

        sync.register(SyncForm.VECTOR, failing_writer)
        result = sync.write("hello", {"user_id": "alice"}, targets=[SyncForm.VECTOR])
        assert not result.all_succeeded
        assert len(result.compensations) == 1
        assert result.compensations[0].form == SyncForm.VECTOR

    def test_retry_compensations_succeeds(self) -> None:
        sync = SourceSynchronizer()
        call_count = [0]

        def flaky_writer(content: str, meta: dict) -> str:
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient failure")
            return f"vec-{content[:3]}"

        sync.register(SyncForm.VECTOR, flaky_writer)
        sync.write("hello", {"user_id": "alice"}, targets=[SyncForm.VECTOR])
        assert sync.pending_compensations == 1
        results = sync.retry_compensations()
        assert len(results) == 1
        assert results[0].success

    def test_retry_compensations_max_retries(self) -> None:
        sync = SourceSynchronizer()

        def always_fails(content: str, meta: dict) -> str:
            raise RuntimeError("permanent failure")

        sync.register(SyncForm.VECTOR, always_fails)
        sync.write("hello", {"user_id": "alice"}, targets=[SyncForm.VECTOR])
        # Retry 4 times (max_retries=3)
        for _ in range(4):
            sync.retry_compensations()
        # After max retries, compensation is discarded
        assert sync.pending_compensations == 0

    def test_no_writer_for_form(self) -> None:
        sync = SourceSynchronizer()
        result = sync.write("hello", {}, targets=[SyncForm.GRAPH])
        assert not result.all_succeeded
        assert result.writes[0].error is not None
        assert "no writer" in result.writes[0].error

    def test_reconcile(self) -> None:
        sync = SourceSynchronizer()
        sync.register(SyncForm.VECTOR, lambda c, m: "vec-id")
        assert sync.reconcile("mem-123")

    def test_reconcile_no_writers(self) -> None:
        sync = SourceSynchronizer()
        assert not sync.reconcile("mem-123")

    def test_authority_priority(self) -> None:
        assert AUTHORITY_PRIORITY == [SyncForm.GRAPH, SyncForm.FILE, SyncForm.VECTOR]

    def test_sync_result_all_succeeded(self) -> None:
        result = SyncResult(writes=[SyncWriteResult(form=SyncForm.VECTOR, success=True)])
        assert result.all_succeeded

    def test_sync_result_has_failures(self) -> None:
        result = SyncResult(
            writes=[SyncWriteResult(form=SyncForm.VECTOR, success=True)],
            compensations=[CompensationTask(form=SyncForm.FILE, memory_id="x", content="c")],
        )
        assert not result.all_succeeded


# ======================================================================
# DriftDetector
# ======================================================================


class TestDriftDetector:
    def test_no_drift(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {
                SyncForm.GRAPH: "alice likes python",
                SyncForm.FILE: "alice likes python",
                SyncForm.VECTOR: "alice likes python",
            },
        )
        assert not report.has_drift

    def test_content_drift(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {
                SyncForm.GRAPH: "alice likes python",
                SyncForm.VECTOR: "alice likes pythn",  # typo
            },
        )
        assert report.has_drift
        assert SyncForm.VECTOR in report.drifted_forms
        assert report.authority_form == SyncForm.GRAPH

    def test_missing_form_drift(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {
                SyncForm.GRAPH: "alice likes python",
                SyncForm.FILE: None,  # missing
                SyncForm.VECTOR: "alice likes python",
            },
        )
        assert report.has_drift
        assert SyncForm.FILE in report.drifted_forms

    def test_all_missing(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {SyncForm.GRAPH: None, SyncForm.FILE: None, SyncForm.VECTOR: None},
        )
        assert report.has_drift
        assert len(report.drifted_forms) == 3

    def test_authority_graph_over_file(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {
                SyncForm.GRAPH: "from graph",
                SyncForm.FILE: "from file",
            },
        )
        assert report.authority_form == SyncForm.GRAPH
        assert SyncForm.FILE in report.drifted_forms

    def test_authority_file_over_vector(self) -> None:
        detector = DriftDetector()
        report = detector.check_memory(
            "mem-1",
            {
                SyncForm.FILE: "from file",
                SyncForm.VECTOR: "from vector",
            },
        )
        assert report.authority_form == SyncForm.FILE
        assert SyncForm.VECTOR in report.drifted_forms

    def test_check_consistency(self) -> None:
        detector = DriftDetector()
        report = detector.check_consistency(
            {
                "mem-1": {
                    SyncForm.GRAPH: "hello",
                    SyncForm.VECTOR: "hello",
                },
                "mem-2": {
                    SyncForm.GRAPH: "world",
                    SyncForm.VECTOR: "wrold",  # drift
                },
                "mem-3": {
                    SyncForm.GRAPH: "foo",
                    SyncForm.VECTOR: "foo",
                },
            }
        )
        assert report.total_checked == 3
        assert report.drift_count == 1
        assert report.consistency_rate == pytest.approx(2 / 3)

    def test_check_consistency_empty(self) -> None:
        detector = DriftDetector()
        report = detector.check_consistency({})
        assert report.total_checked == 0
        assert report.consistency_rate == 1.0

    def test_drift_report_dataclass(self) -> None:
        report = DriftReport(memory_id="mem-1")
        assert not report.has_drift

    def test_consistency_report_dataclass(self) -> None:
        report = ConsistencyReport(total_checked=10, drift_count=2)
        assert report.consistency_rate == pytest.approx(0.8)
