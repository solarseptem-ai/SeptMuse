"""Shim — 向后兼容, RecallPipeline 已迁移到 septmuse.pipelines.recall."""

from septmuse.pipelines.recall import RecallPipeline

__all__ = ["RecallPipeline"]
