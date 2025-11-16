"""
Reasoning Head Discovery Framework

This framework identifies and evaluates reasoning heads used by models for 
backward-chaining style reasoning, integrating with the testing benchmarks.
"""

from .discovery import ReasoningHeadDiscovery
from .subtask_extraction import discover_subtasks
from .head_scoring import HeadScorer, AblationScorer, CausalPatchingScorer, MutualInfoScorer
from .evaluation import ReasoningHeadEvaluator
from .reporting import generate_evaluation_report

__all__ = [
    "ReasoningHeadDiscovery",
    "discover_subtasks",
    "HeadScorer",
    "AblationScorer",
    "CausalPatchingScorer",
    "MutualInfoScorer",
    "ReasoningHeadEvaluator",
    "generate_evaluation_report",
]

