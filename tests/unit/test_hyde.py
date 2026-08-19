"""HyDE 假设文档嵌入单元测试。"""

from unittest.mock import MagicMock


def test_generate_hypothetical_document_with_llm():
    """有 LLM 时应生成假设文档。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.return_value = "张三是公司CTO，负责技术架构决策。"
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result == "张三是公司CTO，负责技术架构决策。"
    assert llm.complete.call_count == 1


def test_generate_hypothetical_document_no_llm():
    """无 LLM 时应返回 None。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    result = generate_hypothetical_document("张三负责什么", None)
    assert result is None


def test_generate_hypothetical_document_llm_error():
    """LLM 异常时应返回 None（降级，不崩溃）。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.side_effect = Exception("Ollama connection refused")
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result is None
