"""
Comprehensive Unit and Integration Test Suite for Phase 5 Calibration Framework.

Verifies:
1. Schemas & Correctness Evaluator (Strict, Fractional, Lenient, 3-state rubric).
2. Calibration Metrics & Murphy Decomposition (BS == REL - RES + UNC invariant).
3. Quantile ECE & MCE computation.
4. Eligibility Filters & Exact Intersection Formulas (Table A & Table B).
5. Standardizer & Recalibration Models 1-4 (Hessian W_l formula, P_val model selection).
6. Hamilton-Hare Largest-Remainder Stratified Random QC Allocation.
7. Two-Tier Multiplicity Bonferroni Bootstrap Testing (Level A & Level B).
"""

import math
import numpy as np
import pytest

from src.calibration.schemas import (
    ClaimState,
    ClaimVerification,
    AnswerVerification,
    CorrectnessEvaluator,
)
from src.calibration.metrics import (
    compute_brier_score,
    compute_murphy_decomposition,
    compute_quantile_ece,
)
from src.calibration.eligibility import (
    EligibilityEngine,
    RecordEligibility,
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
    HamiltonHareAllocator,
    BootstrapInference,
    MultiplicityController,
)


# ==========================================
# 1. Correctness & Claim Verification Tests
# ==========================================

def test_claim_verification_rubric():
    c_sup = ClaimVerification("C1", "Fact 1", ClaimState.SUPPORTED)
    c_ref = ClaimVerification("C2", "Fact 2", ClaimState.REFUTED)
    c_unc = ClaimVerification("C3", "Fact 3", ClaimState.UNCERTAIN)

    assert c_sup.binary_score == 1
    assert c_ref.binary_score == 0
    assert c_unc.binary_score == 0


def test_answer_correctness_evaluator_strict():
    # All SUPPORTED -> Strict = 1, Frac = 1.0, Lenient = 1
    ans_all_sup = CorrectnessEvaluator.evaluate_answer(
        "A1",
        [
            {"claim_id": "c1", "proposition": "p1", "state": "SUPPORTED"},
            {"claim_id": "c2", "proposition": "p2", "state": "SUPPORTED"},
        ],
    )
    assert ans_all_sup.strict_correctness == 1
    assert ans_all_sup.fractional_correctness == 1.0
    assert ans_all_sup.lenient_correctness == 1

    # One UNCERTAIN -> Strict = 0, Frac = 0.5, Lenient = 1
    ans_one_unc = CorrectnessEvaluator.evaluate_answer(
        "A2",
        [
            {"claim_id": "c1", "proposition": "p1", "state": "SUPPORTED"},
            {"claim_id": "c2", "proposition": "p2", "state": "UNCERTAIN"},
        ],
    )
    assert ans_one_unc.strict_correctness == 0
    assert ans_one_unc.fractional_correctness == 0.5
    assert ans_one_unc.lenient_correctness == 1

    # One REFUTED -> Strict = 0, Frac = 0.5, Lenient = 0
    ans_one_ref = CorrectnessEvaluator.evaluate_answer(
        "A3",
        [
            {"claim_id": "c1", "proposition": "p1", "state": "SUPPORTED"},
            {"claim_id": "c2", "proposition": "p2", "state": "REFUTED"},
        ],
    )
    assert ans_one_ref.strict_correctness == 0
    assert ans_one_ref.fractional_correctness == 0.5
    assert ans_one_ref.lenient_correctness == 0


# ==========================================
# 2. Calibration Metrics & Invariant Tests
# ==========================================

def test_brier_score_and_murphy_decomposition_invariant():
    np.random.seed(42)
    conf = np.random.uniform(0.0, 1.0, size=200)
    # Synthetic noisy binary labels
    y = (conf + np.random.normal(0, 0.2, size=200) > 0.5).astype(float)

    decomp = compute_murphy_decomposition(conf, y, num_bins=5)
    bs = compute_brier_score(conf, y)

    assert math.isclose(decomp.brier_score, bs, abs_tol=1e-6)
    # Fundamental Invariant: BS == REL - RES + UNC + RESID
    assert decomp.is_consistent
    assert math.isclose(
        bs,
        decomp.reliability - decomp.resolution + decomp.uncertainty + decomp.bin_covariance_residual,
        abs_tol=1e-5,
    )

    # For discrete binned forecasts c_i = c_bar_b, residual is zero
    discrete_conf = np.zeros_like(conf)
    # Quantile bins
    quantiles = np.linspace(0, 100, 6)
    edges = np.percentile(conf, quantiles)
    edges[0] = -1e-9; edges[-1] = 1.0 + 1e-9
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 4)
    for b in range(5):
        m = (idx == b)
        if np.sum(m) > 0:
            discrete_conf[m] = np.mean(conf[m])
    decomp_discrete = compute_murphy_decomposition(discrete_conf, y, num_bins=5)
    assert math.isclose(decomp_discrete.bin_covariance_residual, 0.0, abs_tol=1e-5)
    assert math.isclose(
        decomp_discrete.brier_score,
        decomp_discrete.reliability - decomp_discrete.resolution + decomp_discrete.uncertainty,
        abs_tol=1e-5,
    )


def test_quantile_ece_and_mce():
    # Perfectly calibrated synthetic predictions
    conf = np.array([0.1] * 20 + [0.9] * 20)
    y = np.array([0.0] * 18 + [1.0] * 2 + [1.0] * 18 + [0.0] * 2)

    metrics = compute_quantile_ece(conf, y, num_bins=2)
    assert metrics.num_samples == 40
    assert metrics.ece >= 0.0
    assert metrics.mce >= metrics.ece
    assert metrics.decomposition.is_consistent


# ==========================================
# 3. Eligibility Filters & Attrition Tests
# ==========================================

def test_eligibility_engine_exact_intersection():
    records = [
        # Question 1: EN ok, HI ok
        {"question_id": "Q1", "condition": "en", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
        {"question_id": "Q1", "condition": "hi", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
        # Question 2: EN ok, HI verb invalid
        {"question_id": "Q2", "condition": "en", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
        {"question_id": "Q2", "condition": "hi", "is_strict_correctness_eligible": True, "is_verbalized_valid": False, "is_statistical_valid": True},
        # Question 3: EN strict invalid, HI ok
        {"question_id": "Q3", "condition": "en", "is_strict_correctness_eligible": False, "is_verbalized_valid": True, "is_statistical_valid": True},
        {"question_id": "Q3", "condition": "hi", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
        # Question 4: Both ok
        {"question_id": "Q4", "condition": "en", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
        {"question_id": "Q4", "condition": "hi", "is_strict_correctness_eligible": True, "is_verbalized_valid": True, "is_statistical_valid": True},
    ]

    engine = EligibilityEngine(records)
    all_qids = ["Q1", "Q2", "Q3", "Q4"]

    # EN self-paired verbalized: Q1, Q2, Q4 (Q3 failed EN strict) -> 3
    en_paired_verb = engine.get_paired_cohort_verbalized("en", all_qids)
    assert set(en_paired_verb) == {"Q1", "Q2", "Q4"}

    # HI vs EN paired verbalized:
    # Q1: EN ok, HI ok -> YES
    # Q2: EN ok, HI verb False -> NO
    # Q3: EN strict False -> NO
    # Q4: EN ok, HI ok -> YES
    hi_paired_verb = engine.get_paired_cohort_verbalized("hi", all_qids)
    assert set(hi_paired_verb) == {"Q1", "Q4"}

    # HI vs EN paired statistical:
    # Q2 has HI stat ok -> YES
    hi_paired_stat = engine.get_paired_cohort_statistical("hi", all_qids)
    assert set(hi_paired_stat) == {"Q1", "Q2", "Q4"}

    # Verify Table B generation
    table_b = engine.compute_table_b("test_partition", all_qids)
    assert len(table_b) == 2  # en and hi
    row_hi = [r for r in table_b if r.condition == "hi"][0]
    assert row_hi.actual_paired_verbalized == 2
    assert row_hi.actual_paired_statistical == 3
    assert row_hi.max_paired_cohort == 4


# ==========================================
# 4. Recalibration Engine & Models 1-4 Tests
# ==========================================

def test_feature_standardization():
    standardizer = Standardizer()
    conf = np.array([0.2, 0.5, 0.8, 0.9])
    standardizer.fit(conf)
    z_tilde = standardizer.transform(conf)

    assert standardizer.is_fitted
    assert len(z_tilde) == 4
    # Mean should be 0.0, std should be 1.0 (ddof=0)
    assert math.isclose(float(np.mean(z_tilde)), 0.0, abs_tol=1e-5)
    assert math.isclose(float(np.std(z_tilde, ddof=0)), 1.0, abs_tol=1e-5)


def test_model_1_and_model_2_fitting_and_selection():
    np.random.seed(123)
    n = 100
    conf = np.random.uniform(0.1, 0.9, size=n)
    conds = ["en"] * 50 + ["ta"] * 50
    # True labels with language shift: Tamil has lower correctness for same confidence
    y = np.zeros(n)
    for i in range(n):
        shift = -1.0 if conds[i] == "ta" else 0.0
        p = 1.0 / (1.0 + np.exp(-(2.0 * conf[i] + shift)))
        y[i] = 1.0 if np.random.rand() < p else 0.0

    engine = RecalibrationEngine()
    engine.fit_on_calibration(conf, y, conds)

    assert engine.model_1.is_fitted
    assert engine.model_2.is_fitted
    assert engine.model_2.deltas["en"] == 0.0  # Identifiability constraint
    assert "ta" in engine.model_2.deltas

    # Test selection on validation partition
    best_model = engine.select_model_on_validation(conf, y, conds)
    assert best_model in ["Model_1_Pooled_Platt", "Model_2_Language_Shift_Platt"]

    preds = engine.predict_recalibrated(conf, conds)
    assert len(preds) == n
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_model_3_hessian_and_exploratory_status():
    np.random.seed(456)
    n = 60
    conf = np.random.uniform(0.2, 0.8, size=n)
    conds = ["en"] * 30 + ["hi"] * 30
    y = np.random.binomial(1, 0.6, size=n)

    m3 = Model3LanguageSpecificPlatt(l2_reg=1.0)
    std = Standardizer().fit(conf)
    z_tilde = std.transform(conf)
    m3.fit(z_tilde, y, conds)

    assert m3.is_fitted
    cov_en = m3.get_parameter_covariance("en")
    cov_hi = m3.get_parameter_covariance("hi")

    assert cov_en is not None and cov_en.shape == (2, 2)
    assert cov_hi is not None and cov_hi.shape == (2, 2)
    # Covariance diagonal must be strictly positive
    assert cov_en[0, 0] > 0.0 and cov_en[1, 1] > 0.0
    assert cov_hi[0, 0] > 0.0 and cov_hi[1, 1] > 0.0


def test_model_4_isotonic_exploratory():
    m4 = Model4Isotonic()
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    m4.fit(z, y)
    assert m4.is_fitted

    preds = m4.predict_proba(np.array([-1.5, 0.5]))
    assert len(preds) == 2
    assert preds[0] <= preds[1]  # Monotonic


# ==========================================
# 5. Statistical Tests & Multiplicity
# ==========================================

def test_hamilton_hare_largest_remainder_allocation():
    # 3 strata with fractional counts summing to 495
    strata = {
        "en->en": 450,
        "hi->hi": 450,
        "ta->ta": 450,
        "bn->bn": 450,
    }
    total_m = 100
    allocations = HamiltonHareAllocator.allocate_sample(strata, total_m)

    assert sum(allocations.values()) == 100
    for s, cnt in allocations.items():
        assert cnt == 25  # Equal split

    # Unequal strata with remainders
    strata_unequal = {
        "stratum_A": 100,  # 100 / 300 * 10 = 3.333 -> 3
        "stratum_B": 100,  # 100 / 300 * 10 = 3.333 -> 3
        "stratum_C": 100,  # 100 / 300 * 10 = 3.333 -> 3 (plus 1 remainder)
    }
    alloc_unequal = HamiltonHareAllocator.allocate_sample(strata_unequal, 10)
    assert sum(alloc_unequal.values()) == 10
    # Top remainder gets the extra unit
    assert max(alloc_unequal.values()) == 4
    assert min(alloc_unequal.values()) == 3


def test_bootstrap_paired_inference_multiplicity_tiers():
    qids = [f"Q_{i:03d}" for i in range(45)]

    # Mock metric difference: always returns +0.15 on average
    def mock_metric_fn(resampled_qids):
        # Deterministic pseudo metric based on hash
        return 0.15 + 0.01 * (len(resampled_qids) % 3)

    # Test Level A (K=5, alpha=0.01)
    res_a = BootstrapInference.test_paired_difference(
        qids,
        mock_metric_fn,
        fwer_tier="Level_A_Within_Track",
        n_bootstraps=500,
    )
    assert res_a.coverage == 0.99
    assert res_a.alpha_comp == 0.01
    assert res_a.ci_lower > 0.0  # Should be significant
    assert res_a.is_significant

    # Test Level B (K=20, alpha=0.0025)
    res_b = BootstrapInference.test_paired_difference(
        qids,
        mock_metric_fn,
        fwer_tier="Level_B_Global",
        n_bootstraps=500,
    )
    assert res_b.coverage == 0.9975
    assert res_b.alpha_comp == 0.0025
    assert res_b.ci_lower > 0.0
    assert res_b.is_significant


# ==========================================
# 6. Critical Boundary & Edge Cases
# ==========================================

def test_empty_and_single_sample_metric_boundaries():
    # Empty inputs
    empty_conf = np.array([])
    empty_y = np.array([])
    assert compute_brier_score(empty_conf, empty_y) == 0.0

    decomp_empty = compute_murphy_decomposition(empty_conf, empty_y)
    assert decomp_empty.brier_score == 0.0
    assert decomp_empty.uncertainty == 0.0
    assert decomp_empty.reliability == 0.0
    assert decomp_empty.resolution == 0.0
    assert decomp_empty.bin_covariance_residual == 0.0
    assert decomp_empty.is_consistent

    metrics_empty = compute_quantile_ece(empty_conf, empty_y)
    assert metrics_empty.num_samples == 0
    assert metrics_empty.ece == 0.0
    assert metrics_empty.mce == 0.0

    # Single sample
    single_conf = np.array([0.95])
    single_y = np.array([1.0])
    metrics_single = compute_quantile_ece(single_conf, single_y, num_bins=1)
    assert metrics_single.num_samples == 1
    assert math.isclose(metrics_single.brier_score, (0.95 - 1.0) ** 2, abs_tol=1e-6)
    assert metrics_single.decomposition.is_consistent


def test_extreme_confidence_boundaries_and_clipping():
    # Extreme values: 0.0 and 1.0
    conf = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    std = Standardizer()
    std.fit(conf)
    z = std.transform(conf)
    # Must produce finite floats, no NaN or Inf
    assert np.all(np.isfinite(z))

    # Constant confidence profile (zero variance)
    const_conf = np.array([0.8, 0.8, 0.8, 0.8])
    std_const = Standardizer()
    std_const.fit(const_conf)
    z_const = std_const.transform(const_conf)
    assert np.all(np.isfinite(z_const))


def test_answer_verification_empty_and_ambiguous_edge_cases():
    # Zero claims
    ans_empty = CorrectnessEvaluator.evaluate_answer("A_EMPTY", [])
    assert ans_empty.claim_count == 0
    assert ans_empty.strict_correctness == 0
    assert ans_empty.fractional_correctness == 0.0
    assert ans_empty.lenient_correctness == 0

    # Only UNCERTAIN claims
    ans_all_unc = CorrectnessEvaluator.evaluate_answer(
        "A_UNC",
        [
            {"claim_id": "c1", "proposition": "p1", "state": "UNCERTAIN"},
            {"claim_id": "c2", "proposition": "p2", "state": "UNCERTAIN"},
        ],
    )
    assert ans_all_unc.strict_correctness == 0
    assert ans_all_unc.fractional_correctness == 0.0
    assert ans_all_unc.lenient_correctness == 0

    # Mixed with unknown state string defaults to UNCERTAIN
    ans_mixed = CorrectnessEvaluator.evaluate_answer(
        "A_MIXED",
        [
            {"claim_id": "c1", "proposition": "p1", "state": "SUPPORTED"},
            {"claim_id": "c2", "proposition": "p2", "state": "INVALID_STATE_STRING"},
        ],
    )
    assert ans_mixed.claims[1].state == ClaimState.UNCERTAIN
    assert ans_mixed.strict_correctness == 0
    assert ans_mixed.fractional_correctness == 0.5
    assert ans_mixed.lenient_correctness == 1


def test_hamilton_hare_zero_and_saturated_sample_bounds():
    strata = {"A": 10, "B": 20}
    # Zero sample requested
    assert HamiltonHareAllocator.allocate_sample(strata, 0) == {"A": 0, "B": 0}
    # Saturated sample requested (sample >= total)
    assert HamiltonHareAllocator.allocate_sample(strata, 30) == {"A": 10, "B": 20}
    assert HamiltonHareAllocator.allocate_sample(strata, 50) == {"A": 10, "B": 20}
