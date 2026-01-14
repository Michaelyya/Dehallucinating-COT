"""
Reasoning Head Discovery Framework

This framework identifies and evaluates reasoning heads used by models for
backward-chaining style reasoning, integrating with the testing benchmarks.

=============================================================================
EFFICIENT SCORING METHODS (NEW)
=============================================================================

The framework now includes efficient alternatives to brute-force ablation:

1. ActivationScorer - Activation-based scoring (~1000x faster)
2. GradientScorer - Gradient-based scoring (~500x faster)
3. AttentionPatternScorer - Attention pattern analysis (~1000x faster)
4. CombinedScorer - Multi-method combination (~300x faster)

Use create_efficient_scorer() to instantiate these scorers:

    from reasoning_heads import create_efficient_scorer

    scorer = create_efficient_scorer("activation", model, tokenizer, device)
    scores = scorer.score_all_heads(examples, subtask_name)

Or use the new discovery script:

    python discover_reasoning_heads_efficient.py --method activation

=============================================================================
"""

from .discovery import ReasoningHeadDiscovery
from .subtask_extraction import discover_subtasks
from .head_scoring import HeadScorer, AblationScorer, CausalPatchingScorer, MutualInfoScorer, create_scorer
from .evaluation import ReasoningHeadEvaluator
from .reporting import generate_evaluation_report

# Import efficient scorers
try:
    from .efficient_head_scoring import (
        EfficientHeadScorer,
        ActivationScorer,
        GradientScorer,
        AttentionPatternScorer,
        CombinedScorer,
        create_efficient_scorer,
    )
    EFFICIENT_SCORERS_AVAILABLE = True
except ImportError:
    EFFICIENT_SCORERS_AVAILABLE = False
    # Define stubs for compatibility
    EfficientHeadScorer = None
    ActivationScorer = None
    GradientScorer = None
    AttentionPatternScorer = None
    CombinedScorer = None
    create_efficient_scorer = None

__all__ = [
    # Core discovery
    "ReasoningHeadDiscovery",
    "discover_subtasks",
    # Original scorers
    "HeadScorer",
    "AblationScorer",
    "CausalPatchingScorer",
    "MutualInfoScorer",
    "create_scorer",
    # Efficient scorers (new)
    "EfficientHeadScorer",
    "ActivationScorer",
    "GradientScorer",
    "AttentionPatternScorer",
    "CombinedScorer",
    "create_efficient_scorer",
    "EFFICIENT_SCORERS_AVAILABLE",
    # Evaluation
    "ReasoningHeadEvaluator",
    "generate_evaluation_report",
]

