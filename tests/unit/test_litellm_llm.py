"""litellm LLM provider 测试。"""

from unittest.mock import MagicMock, patch

from septmuse.llms.litellm import LitellmLLM


def test_litellm_complete_delegates():
    """complete 委托 litellm.completion。"""
    with patch("litellm.completion") as mock_completion:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="hello world"))]
        mock_completion.return_value = mock_resp

        llm = LitellmLLM(model="gpt-4o-mini", api_key="test-key")
        result = llm.complete("system prompt", "user prompt")

        assert result == "hello world"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["api_key"] == "test-key"


def test_litellm_config():
    """LitellmLLMConfig 基本字段。"""
    from septmuse.configs.llms.litellm import LitellmLLMConfig

    config = LitellmLLMConfig(model="groq/llama-3.1-70b-versatile", api_key="test")
    assert config.backend == "litellm"
    assert config.model == "groq/llama-3.1-70b-versatile"
