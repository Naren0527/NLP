"""
Phase 5 Calibration Framework Package.

Implements:
1. Verification Schemas & Correctness Evaluator (Claim decomposition, 3-state rubric, strict/frac/lenient)
2. Calibration Metrics Engine (Brier score, Murphy decomposition, Quantile ECE/MCE/RMSCE)
3. Eligibility Filters & Attrition Accounting (Tables A & B, record-level intersections)
4. Recalibration Engine (Feature standardization, Models 1-4, Model selection on P_val)
5. Statistical Inference & Multiplicity Control (Level A/B Bonferroni bootstrap CIs, Hamilton-Hare QC)
"""

from src.calibration.schemas import (
    ClaimState,
    ClaimVerification,
    AnswerVerification,
    CorrectnessEvaluator,
)
from src.calibration.metrics import (
    CalibrationMetrics,
    compute_brier_score,
    compute_murphy_decomposition,
    compute_quantile_ece,
)
from src.calibration.eligibility import (
    EligibilityEngine,
    RecordEligibility,
    AttritionSummary,
)
from src.calibration.recalibration import (
    Standardizer,
    Model1PooledPlatt,
    Model2LanguageShiftPlatt,
    Model3LanguageSpecificPlatt,
    Model4Isotonic,
    RecalibrationEngine,
)
from src.calibration.statistical_tests import (
    BootstrapInference,
    HamiltonHareAllocator,
    MultiplicityController,
)

__all__ = [
    "ClaimState",
    "ClaimVerification",
    "AnswerVerification",
    "CorrectnessEvaluator",
    "CalibrationMetrics",
    "compute_brier_score",
    "compute_murphy_decomposition",
    "compute_quantile_ece",
    "EligibilityEngine",
    "RecordEligibility",
    "AttritionSummary",
    "Standardizer",
    "Model1PooledPlatt",
    "Model2LanguageShiftPlatt",
    "Model3LanguageSpecificPlatt",
    "Model4Isotonic",
    "RecalibrationEngine",
    "BootstrapInference",
    "HamiltonHareAllocator",
    "MultiplicityController",
]
