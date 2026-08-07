"""HyDE 假设文档检索 prompt — LLM 生成假设答案用于检索."""

HYDE_PROMPT = """Given the query, generate a hypothetical answer (2-3 sentences) that would be a good match for retrieving relevant memories. Write what the stored memory would look like, not a direct answer to the question.

Query: {query}

Hypothetical memory text:"""
