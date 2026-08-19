#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""septmuse.domains — Memory facade 拆分后的领域模块。

5 个领域模块:
- SearchModule: 检索 (search / search_hybrid / search_graph / ...)
- EvolutionModule: 演化 (improve / reflect / dream / 自进化 / ...)
- CaptureModule: 捕获/写入 (remember / capture / add_typed / ...)
- EntityModule: 实体 (search_entities / list_entities / ...)
- TemporalModule: 时态 (search_at / search_interval / ...)

元认知通过 MetacognitionLayer (memory/meta.py) 访问，不再需要独立领域模块。
"""

from septmuse.domains.capture import CaptureModule
from septmuse.domains.entities import EntityModule
from septmuse.domains.evolution import EvolutionModule
from septmuse.domains.search import SearchModule
from septmuse.domains.temporal import TemporalModule

__all__ = [
    "CaptureModule",
    "EntityModule",
    "EvolutionModule",
    "SearchModule",
    "TemporalModule",
]
