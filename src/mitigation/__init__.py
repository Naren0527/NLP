"""
Mitigation module initialization.
"""
from src.mitigation.strategies import (
    BaseMitigationStrategy,
    StrategyA_NativePromptOptimization,
    StrategyB_CrossLingualPivotAnswering,
    StrategyC_BilingualVerification,
    StrategyD_EvidenceGroundedRefinement,
    StrategyE_SelectiveAnswering,
    MitigationOutput,
    LANG_CONFIGS,
)

__all__ = [
    "BaseMitigationStrategy",
    "StrategyA_NativePromptOptimization",
    "StrategyB_CrossLingualPivotAnswering",
    "StrategyC_BilingualVerification",
    "StrategyD_EvidenceGroundedRefinement",
    "StrategyE_SelectiveAnswering",
    "MitigationOutput",
    "LANG_CONFIGS",
]
