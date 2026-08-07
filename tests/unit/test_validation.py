"""输入校验工具测试 (对齐 mem0 _validate_and_trim_entity_id 等)."""

import pytest

from septmuse.core.validation import (
    validate_entity_id,
    validate_search_params,
    validate_search_query,
)

# ======================== validate_entity_id ========================


class TestValidateEntityId:
    def test_normal_string(self):
        assert validate_entity_id("alice") == "alice"

    def test_trim_whitespace(self):
        assert validate_entity_id("  alice  ") == "alice"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_entity_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_entity_id("   ")

    def test_internal_whitespace_raises(self):
        with pytest.raises(ValueError, match="internal whitespace"):
            validate_entity_id("al ice")

    def test_none_returns_none(self):
        assert validate_entity_id(None) is None

    def test_non_string_coerced(self):
        assert validate_entity_id(123) == "123"

    def test_custom_name_in_error(self):
        with pytest.raises(ValueError, match="user_id"):
            validate_entity_id("", name="user_id")

    def test_tab_internal_whitespace_raises(self):
        with pytest.raises(ValueError, match="internal whitespace"):
            validate_entity_id("al\tice")


# ======================== validate_search_params ========================


class TestValidateSearchParams:
    def test_valid_params(self):
        validate_search_params(threshold=0.5, top_k=10)

    def test_none_params_ok(self):
        validate_search_params(threshold=None, top_k=None)

    def test_threshold_zero_ok(self):
        validate_search_params(threshold=0.0)

    def test_threshold_one_ok(self):
        validate_search_params(threshold=1.0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="Must be between 0 and 1"):
            validate_search_params(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="Must be between 0 and 1"):
            validate_search_params(threshold=1.5)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="Must be non-negative"):
            validate_search_params(top_k=-1)

    def test_top_k_zero_ok(self):
        validate_search_params(top_k=0)

    def test_top_k_bool_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            validate_search_params(top_k=True)

    def test_threshold_bool_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_search_params(threshold=True)


# ======================== validate_search_query ========================


class TestValidateSearchQuery:
    def test_normal_query(self):
        assert validate_search_query("hello") == "hello"

    def test_trim_query(self):
        assert validate_search_query("  hello world  ") == "hello world"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_search_query("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_search_query("   ")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_search_query(123)

    def test_non_string_none_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_search_query(None)
