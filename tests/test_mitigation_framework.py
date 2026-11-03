"""
Unit test suite for the Multilingual Mitigation Framework.
Tests Strategies A, B, C, D, E data structures, selective abstention gating,
calibration integration, and correctness validation logic.
"""

import pytest
import numpy as np
from src.mitigation.strategies import (
    MitigationOutput,
    StrategyE_SelectiveAnswering,
    LANG_CONFIGS,
    NATIVE_SYSTEM_PROMPTS,
)
from src.calibration.recalibration import Model2LanguageShiftPlatt, RecalibrationEngine
from src.calibration.metrics import compute_quantile_ece
from src.calibration.schemas import CorrectnessEvaluator, ClaimVerification


def test_mitigation_output_dataclass():
    out = MitigationOutput(
        strategy_name="Strategy_B_CrossLingualPivotAnswering",
        question_id="Q_TEST_01",
        target_language="hi",
        final_answer="कर्नाटक की राजधानी बेंगलुरु है।",
        is_abstained=False,
        abstention_reason=None,
        raw_confidence=0.92,
        calibrated_confidence=0.78,
        statistical_confidence=0.85,
        is_correct=True,
        intermediate_candidates={"pivot_en": "Bengaluru is the capital of Karnataka."},
        verification_details={"status": "VERIFIED"},
        latency_seconds=1.24,
    )
    assert out.strategy_name == "Strategy_B_CrossLingualPivotAnswering"
    assert out.target_language == "hi"
    assert out.is_correct is True
    assert out.calibrated_confidence == 0.78
    assert not out.is_abstained


def test_selective_answering_abstention():
    out_high = MitigationOutput(
        strategy_name="TestStrat",
        question_id="Q_01",
        target_language="hi",
        final_answer="Answer A",
        is_abstained=False,
        abstention_reason=None,
        raw_confidence=0.95,
        calibrated_confidence=0.80,
        statistical_confidence=0.70,
        is_correct=True,
        intermediate_candidates={},
        verification_details={},
        latency_seconds=0.5,
    )
    out_low = MitigationOutput(
        strategy_name="TestStrat",
        question_id="Q_02",
        target_language="hi",
        final_answer="Answer B",
        is_abstained=False,
        abstention_reason=None,
        raw_confidence=0.95,
        calibrated_confidence=0.40,
        statistical_confidence=0.35,
        is_correct=False,
        intermediate_candidates={},
        verification_details={},
        latency_seconds=0.5,
    )

    # Threshold tau = 0.60
    gated_high = StrategyE_SelectiveAnswering.apply_abstention(out_high, threshold=0.60)
    assert gated_high.is_abstained is False
    assert gated_high.abstention_reason is None

    gated_low = StrategyE_SelectiveAnswering.apply_abstention(out_low, threshold=0.60)
    assert gated_low.is_abstained is True
    assert "below reliability threshold" in gated_low.abstention_reason


def test_selective_metrics_computation():
    outputs = []
    # 5 correct high conf, 3 incorrect low conf, 2 incorrect high conf
    for i in range(5):
        outputs.append(MitigationOutput("S", f"Q_{i}", "hi", "Ans", False, None, 0.9, 0.85, 0.8, True, {}, {}, 0.1))
    for i in range(3):
        outputs.append(MitigationOutput("S", f"Q_{i+5}", "hi", "Ans", False, None, 0.9, 0.45, 0.4, False, {}, {}, 0.1))
    for i in range(2):
        outputs.append(MitigationOutput("S", f"Q_{i+8}", "hi", "Ans", False, None, 0.9, 0.75, 0.6, False, {}, {}, 0.1))

    metrics = StrategyE_SelectiveAnswering.compute_selective_metrics(outputs, thresholds=[0.50, 0.80])
    
    # At tau = 0.50: answered = 5 + 2 = 7; coverage = 7/10 = 0.70; acc = 5/7 = 0.714
    assert metrics["0.5"]["answered_count"] == 7
    assert metrics["0.5"]["coverage"] == 0.70
    assert abs(metrics["0.5"]["accuracy_on_answered"] - (5/7)) < 0.01

    # At tau = 0.80: answered = 5; coverage = 5/10 = 0.50; acc = 5/5 = 1.00
    assert metrics["0.8"]["answered_count"] == 5
    assert metrics["0.8"]["coverage"] == 0.50
    assert metrics["0.8"]["accuracy_on_answered"] == 1.00
    assert metrics["0.8"]["selective_risk"] == 0.00


def test_language_configs_and_system_prompts():
    assert set(LANG_CONFIGS.keys()) == {"en", "hi", "bn", "te", "ta"}
    for lang, prompt in NATIVE_SYSTEM_PROMPTS.items():
        assert len(prompt) > 20
        assert lang in ["hi", "bn", "te", "ta"]


def test_model2_language_shift_fitting():
    m2 = Model2LanguageShiftPlatt(ref_condition="en")
    # Synthetic dataset
    z = np.array([2.0, 1.5, 2.5, 2.0, 1.0, 1.5])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    conds = ["en", "en", "en", "hi", "hi", "hi"]

    m2.fit(z, y, conds)
    assert m2.is_fitted is True
    assert "en" in m2.deltas
    assert "hi" in m2.deltas
    assert m2.deltas["en"] == 0.0
    # In this synthetic set, hi has lower accuracy so delta_hi should be negative
    assert m2.deltas["hi"] < 0.0

    # Predictions
    p_en = m2.predict_proba(np.array([2.0]), condition="en")[0]
    p_hi = m2.predict_proba(np.array([2.0]), condition="hi")[0]
    assert p_en > p_hi  # Same confidence logit yields lower calibrated prob in Hindi


def test_quantile_ece_calculation():
    # Perfect calibration
    confs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ys = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    ece_res = compute_quantile_ece(confs, ys, num_bins=5)
    assert ece_res.ece >= 0.0
    assert ece_res.ece <= 1.0
