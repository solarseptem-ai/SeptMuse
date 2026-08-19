"""Pipelines — 统一管线编排层。

所有 Pipeline 从此入口导出, 不再散落在 memory/ / retrieval/ / capture/ 下。
管线构造由 ComponentRegistry (memory/registry_v2.py) 统一管理。
"""

from septmuse.pipelines.add import AddPipeline
from septmuse.pipelines.capture import CapturePipeline, PipelineResult, PreprocessResult
from septmuse.pipelines.forget import ForgetPipeline
from septmuse.pipelines.recall import RecallPipeline
from septmuse.pipelines.remember import RememberPipeline
from septmuse.pipelines.reset import ResetPipeline
from septmuse.pipelines.search import SearchPipeline, VectorRetrieverAdapter

__all__ = [
    "AddPipeline",
    "CapturePipeline",
    "ForgetPipeline",
    "PipelineResult",
    "PreprocessResult",
    "RecallPipeline",
    "RememberPipeline",
    "ResetPipeline",
    "SearchPipeline",
    "VectorRetrieverAdapter",
]
