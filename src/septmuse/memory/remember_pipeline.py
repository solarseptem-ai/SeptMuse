"""Shim — 向后兼容, RememberPipeline 已迁移到 septmuse.pipelines.remember."""

from septmuse.pipelines.remember import RememberPipeline

__all__ = ["RememberPipeline"]
