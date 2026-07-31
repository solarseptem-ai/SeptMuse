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
"""阶段3 Batch1 治理模块单元测试 — token_budget / privacy / approval / degradation。

固化 (架构文档 §5.3 治理):
- TokenBudget: 贪心裁剪, 高分优先, 超预算截断
- PrivacyFilter: regex 脱敏, API keys/Bearer/密码/信用卡
- WriteValidator: 参数校验 + SHA-256 hash 去重 + 时间窗
- DegradationPolicy: Cass helpful/harmful 退化, 独立可复用
"""

from __future__ import annotations

import pytest

from septmuse.governance.approval import (
    DedupWindow,
    WriteValidator,
    compute_hash,
    validate_entity_id,
)
from septmuse.governance.degradation import DegradationPolicy, DegradationRecord
from septmuse.governance.privacy import PrivacyFilter
from septmuse.governance.token_budget import (
    DEFAULT_TOKEN_BUDGET,
    BudgetItem,
    TokenBudget,
    estimate_tokens,
)

# ======================================================================
# TokenBudget
# ======================================================================


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short(self) -> None:
        assert estimate_tokens("abcd") == 1

    def test_long(self) -> None:
        assert estimate_tokens("a" * 400) == 100

    def test_non_string(self) -> None:
        assert estimate_tokens("12345678") == 2


class TestTokenBudget:
    def test_fit_all_under_budget(self) -> None:
        budget = TokenBudget(budget=1000)
        items = [
            BudgetItem(text="a" * 40, score=0.9),
            BudgetItem(text="b" * 40, score=0.5),
        ]
        result = budget.fit(items)
        assert len(result.items) == 2
        assert result.used_tokens == 20
        assert result.dropped == 0

    def test_fit_drops_low_score_when_over_budget(self) -> None:
        budget = TokenBudget(budget=10)
        items = [
            BudgetItem(text="high" * 8, score=0.9),  # 32 chars = 8 tokens, fits
            BudgetItem(text="low" * 8, score=0.1),  # 24 chars = 6 tokens, 8+6=14 > 10, dropped
        ]
        result = budget.fit(items)
        assert len(result.items) == 1
        assert result.items[0].text.startswith("high")
        assert result.dropped == 1

    def test_fit_high_score_first(self) -> None:
        budget = TokenBudget(budget=10)
        items = [
            BudgetItem(text="x" * 40, score=0.1),  # 10 tokens
            BudgetItem(text="y" * 40, score=0.9),  # 10 tokens, fits first (high score)
        ]
        result = budget.fit(items)
        assert len(result.items) == 1
        assert result.items[0].text.startswith("y")

    def test_fit_texts_convenience(self) -> None:
        budget = TokenBudget(budget=100)
        texts = ["hello world", "foo bar baz"]
        result = budget.fit_texts(texts)
        assert len(result) == 2

    def test_fit_texts_with_scores(self) -> None:
        budget = TokenBudget(budget=1)
        texts = ["aaaa", "bbbb"]  # each 1 token, budget=1 → only first (high score) fits
        scores = [0.9, 0.1]
        result = budget.fit_texts(texts, scores)
        assert len(result) == 1
        assert result[0] == "aaaa"

    def test_default_budget(self) -> None:
        assert DEFAULT_TOKEN_BUDGET == 2000

    def test_empty_items(self) -> None:
        budget = TokenBudget(budget=100)
        result = budget.fit([])
        assert len(result.items) == 0
        assert result.used_tokens == 0


# ======================================================================
# PrivacyFilter
# ======================================================================


class TestPrivacyFilter:
    def test_redact_openai_key(self) -> None:
        pf = PrivacyFilter()
        text = "my key is sk-" + "a" * 30
        cleaned = pf.redact(text)
        assert "sk-" not in cleaned
        assert "[REDACTED_OPENAI_KEY]" in cleaned

    def test_redact_aws_key(self) -> None:
        pf = PrivacyFilter()
        text = "aws key AKIA" + "0" * 16
        cleaned = pf.redact(text)
        assert "AKIA" not in cleaned
        assert "[REDACTED_AWS_KEY]" in cleaned

    def test_redact_github_token(self) -> None:
        pf = PrivacyFilter()
        text = "ghp_" + "a" * 36
        cleaned = pf.redact(text)
        assert "ghp_" not in cleaned
        assert "[REDACTED_GITHUB_TOKEN]" in cleaned

    def test_redact_bearer_token(self) -> None:
        pf = PrivacyFilter()
        text = "Authorization: Bearer eyJ" + "a" * 20
        cleaned = pf.redact(text)
        assert "Bearer" not in cleaned
        assert "[REDACTED_BEARER_TOKEN]" in cleaned

    def test_redact_password_assignment(self) -> None:
        pf = PrivacyFilter()
        text = "password=secret123"
        cleaned = pf.redact(text)
        assert "secret123" not in cleaned
        assert "[REDACTED_PASSWORD]" in cleaned

    def test_redact_api_key_generic(self) -> None:
        pf = PrivacyFilter()
        text = "api_key=abcdef0123456789abcd"
        cleaned = pf.redact(text)
        assert "abcdef0123456789" not in cleaned

    def test_redact_credit_card(self) -> None:
        pf = PrivacyFilter()
        text = "card: 4111 1111 1111 1111"
        cleaned = pf.redact(text)
        assert "4111" not in cleaned
        assert "[REDACTED_CC]" in cleaned

    def test_redact_ssn(self) -> None:
        pf = PrivacyFilter()
        text = "ssn: 123-45-6789"
        cleaned = pf.redact(text)
        assert "123-45-6789" not in cleaned
        assert "[REDACTED_SSN]" in cleaned

    def test_no_sensitive(self) -> None:
        pf = PrivacyFilter()
        text = "hello world"
        cleaned = pf.redact(text)
        assert cleaned == text

    def test_has_sensitive_true(self) -> None:
        pf = PrivacyFilter()
        assert pf.has_sensitive("sk-" + "a" * 30)

    def test_has_sensitive_false(self) -> None:
        pf = PrivacyFilter()
        assert not pf.has_sensitive("hello world")

    def test_empty_string(self) -> None:
        pf = PrivacyFilter()
        assert pf.redact("") == ""
        assert not pf.has_sensitive("")

    def test_custom_patterns(self) -> None:
        pf = PrivacyFilter(patterns={"custom": (r"MY-[0-9]+", "[REDACTED_CUSTOM]")})
        text = "id: MY-12345"
        cleaned = pf.redact(text)
        assert "MY-12345" not in cleaned
        assert "[REDACTED_CUSTOM]" in cleaned

    def test_multiple_secrets(self) -> None:
        pf = PrivacyFilter()
        text = f"key=sk-{'a' * 30} and card 4111 1111 1111 1111"
        cleaned = pf.redact(text)
        assert "sk-" not in cleaned
        assert "4111" not in cleaned


# ======================================================================
# WriteValidator / DedupWindow / validate_entity_id
# ======================================================================


class TestValidateEntityId:
    def test_none_returns_none(self) -> None:
        assert validate_entity_id(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert validate_entity_id("") is None

    def test_whitespace_returns_none(self) -> None:
        assert validate_entity_id("   ") is None

    def test_valid_string(self) -> None:
        assert validate_entity_id("alice") == "alice"

    def test_trims_whitespace(self) -> None:
        assert validate_entity_id("  alice  ") == "alice"

    def test_int_converts(self) -> None:
        assert validate_entity_id(123) == "123"

    def test_internal_space_raises(self) -> None:
        with pytest.raises(ValueError, match="internal spaces"):
            validate_entity_id("alice bob")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be str"):
            validate_entity_id([])  # type: ignore[arg-type]


class TestDedupWindow:
    def test_first_add_not_duplicate(self) -> None:
        window = DedupWindow(window_seconds=300)
        assert not window.is_duplicate("hello")

    def test_second_add_is_duplicate(self) -> None:
        window = DedupWindow(window_seconds=300)
        window.add("hello")
        assert window.is_duplicate("hello")

    def test_different_text_not_duplicate(self) -> None:
        window = DedupWindow(window_seconds=300)
        window.add("hello")
        assert not window.is_duplicate("world")

    def test_expired_not_duplicate(self) -> None:
        window = DedupWindow(window_seconds=1)
        window.add("hello")
        # 手动设置过期时间戳 (避免 Windows monotonic 分辨率问题)
        import time

        old_ts = time.monotonic() - 2
        for k in list(window._seen):
            window._seen[k] = old_ts
        assert not window.is_duplicate("hello")

    def test_clear(self) -> None:
        window = DedupWindow(window_seconds=300)
        window.add("hello")
        window.clear()
        assert not window.is_duplicate("hello")


class TestWriteValidator:
    def test_valid_write(self) -> None:
        v = WriteValidator()
        result = v.validate("hello", user_id="alice")
        assert result.allowed
        assert not result.dedup
        assert len(result.text_hash) == 64  # SHA-256 hex

    def test_empty_text_rejected(self) -> None:
        v = WriteValidator()
        result = v.validate("", user_id="alice")
        assert not result.allowed
        assert "empty" in result.reason

    def test_no_entity_id_rejected(self) -> None:
        v = WriteValidator()
        result = v.validate("hello")
        assert not result.allowed
        assert "user_id" in result.reason or "agent_id" in result.reason

    def test_duplicate_rejected(self) -> None:
        v = WriteValidator()
        v.validate("hello", user_id="alice")
        result = v.validate("hello", user_id="alice")
        assert not result.allowed
        assert result.dedup

    def test_agent_id_alone_ok(self) -> None:
        v = WriteValidator()
        result = v.validate("hello", agent_id="bot1")
        assert result.allowed

    def test_whitespace_text_rejected(self) -> None:
        v = WriteValidator()
        result = v.validate("   ", user_id="alice")
        assert not result.allowed

    def test_compute_hash_deterministic(self) -> None:
        assert compute_hash("hello") == compute_hash("hello")
        assert compute_hash("hello") != compute_hash("world")


# ======================================================================
# DegradationPolicy
# ======================================================================


class TestDegradationPolicy:
    def test_initial_confidence(self) -> None:
        record = DegradationRecord()
        assert record.confidence == 0.5

    def test_helpful_increases_confidence(self) -> None:
        policy = DegradationPolicy()
        record = DegradationRecord()
        policy.record_outcome(record, helpful=True)
        assert record.helpful_count == 1
        assert record.confidence == 1.0
        assert not record.deprecated

    def test_harmful_decreases_confidence(self) -> None:
        policy = DegradationPolicy()
        record = DegradationRecord()
        policy.record_outcome(record, helpful=False)
        assert record.harmful_count == 1
        assert record.confidence == 0.0
        assert not record.deprecated  # only 1 harmful, below threshold

    def test_deprecation_at_threshold(self) -> None:
        policy = DegradationPolicy(deprecation_threshold=3)
        record = DegradationRecord()
        for _ in range(3):
            policy.record_outcome(record, helpful=False)
        assert record.harmful_count == 3
        assert record.deprecated  # harmful(3) > helpful(0) and harmful >= 3

    def test_no_deprecation_when_helpful_balances(self) -> None:
        policy = DegradationPolicy(deprecation_threshold=3)
        record = DegradationRecord()
        policy.record_outcome(record, helpful=True)
        policy.record_outcome(record, helpful=True)
        policy.record_outcome(record, helpful=True)
        policy.record_outcome(record, helpful=False)
        policy.record_outcome(record, helpful=False)
        policy.record_outcome(record, helpful=False)
        assert record.harmful_count == 3
        assert record.helpful_count == 3
        assert not record.deprecated  # harmful(3) not > helpful(3)

    def test_should_inject_not_deprecated(self) -> None:
        policy = DegradationPolicy()
        record = DegradationRecord()
        assert policy.should_inject(record)

    def test_should_not_inject_deprecated(self) -> None:
        policy = DegradationPolicy()
        record = DegradationRecord(deprecated=True)
        assert not policy.should_inject(record)

    def test_should_inject_with_confidence_threshold(self) -> None:
        policy = DegradationPolicy()
        record = DegradationRecord(helpful_count=1, harmful_count=3)
        # confidence = 1/4 = 0.25
        assert not policy.should_inject_with_confidence(record, min_confidence=0.5)
        assert policy.should_inject_with_confidence(record, min_confidence=0.1)

    def test_recover_from_deprecation(self) -> None:
        """helpful 追上来后应取消废弃 (Cass 不自动恢复, 但 SeptMuse 逻辑: harmful 不再 > helpful)。"""
        policy = DegradationPolicy(deprecation_threshold=3)
        record = DegradationRecord()
        for _ in range(3):
            policy.record_outcome(record, helpful=False)
        assert record.deprecated
        # 追加 4 个 helpful
        for _ in range(4):
            policy.record_outcome(record, helpful=True)
        # helpful(4) > harmful(3), 但 deprecated 标记是单向的 (不自动取消, 对齐 Cass)
        # 注: record_outcome 只在 harmful > helpful 时设 deprecated=True, 不会重置为 False
        # 这是 Cass 的设计: 废弃的规则不再恢复, 应新增规则
        assert record.deprecated  # 仍废弃
