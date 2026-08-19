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
"""记忆演化模块。"""

from septmuse.evolution.catchup import LLMCatchUpQueue
from septmuse.evolution.conflict import ConflictResolver
from septmuse.evolution.degradation import DegradationPolicy, DegradationRecord
from septmuse.evolution.dream import DreamIntegrator, DreamResult
from septmuse.evolution.feedback import RetrievalFeedback, RetrievalFeedbackResult, RuleFeedback
from septmuse.evolution.refine import RefinementCandidate, RefinementResult, RuleRefiner
from septmuse.evolution.reflect import LessonProposal, ReflectionResult, SessionReflector
from septmuse.evolution.scheduler import EvolutionScheduler
from septmuse.evolution.self_loop import EvolutionCycleResult, SelfEvolutionLoop
from septmuse.evolution.summarizer import Summarizer
from septmuse.evolution.zettel import MemoryLink, ZettelLinker

__all__ = [
    "ConflictResolver",
    "DegradationPolicy",
    "DegradationRecord",
    "DreamIntegrator",
    "DreamResult",
    "EvolutionCycleResult",
    "EvolutionScheduler",
    "LLMCatchUpQueue",
    "LessonProposal",
    "MemoryLink",
    "RefinementCandidate",
    "RefinementResult",
    "ReflectionResult",
    "RetrievalFeedback",
    "RetrievalFeedbackResult",
    "RuleFeedback",
    "RuleRefiner",
    "SelfEvolutionLoop",
    "SessionReflector",
    "Summarizer",
    "ZettelLinker",
]
