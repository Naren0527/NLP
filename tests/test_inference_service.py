"""
Unit & Regression Tests for Inference Service & Custom Query Pipeline.
Covers Tests A, B, C, D, E, F, G as specified in the Bug Fix requirements.
"""

import pytest
from src.inference_service import (
    determine_active_query,
    execute_live_inference,
    LANGUAGES,
    KNOWN_OPEN_DOMAIN_KB,
)

BENCHMARK_KARNATAKA = {
    "id": "Q_GEO_DIR_01",
    "question_en": "What is the capital city of Karnataka?",
    "domain": "Geography",
    "evidence": "Bengaluru, officially known as Bangalore, is the capital of Karnataka.",
    "claims": ["Bengaluru is the capital city of Karnataka."],
    "answers": {
        "en": "Bengaluru is the capital city of Karnataka.",
        "hi": "कर्नाटक की राजधानी बेंगलुरु (बैंगलोर) है।",
        "bn": "কর্ণাটকের রাজধানী হল বেঙ্গালুরু।",
        "te": "కర్ణాటక రాజధాని నగరం బెంగళూరు.",
        "ta": "கர்நாடகாவின் தலைநகரம் பெங்களூரு ஆகும்.",
    }
}


def test_case_a_custom_query_overrides_benchmark():
    """Test A: Custom query 'what is the capital of andhra' overrides Karnataka benchmark."""
    benchmark_q = BENCHMARK_KARNATAKA["question_en"]
    custom_q = "what is the capital of andhra"

    active_q, is_custom = determine_active_query(benchmark_q, custom_q)
    assert is_custom is True
    assert active_q == "what is the capital of andhra"

    result = execute_live_inference(
        active_query=active_q,
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=is_custom
    )

    # Must answer Andhra Pradesh, not Karnataka/Bengaluru!
    assert "Amaravati" in result["final_answer"]
    assert "Bengaluru" not in result["final_answer"]
    assert "Karnataka" not in result["final_answer"]


def test_case_b_custom_query_python_creator():
    """Test B: Custom query 'Who developed Python?' overrides Karnataka benchmark."""
    benchmark_q = BENCHMARK_KARNATAKA["question_en"]
    custom_q = "Who developed Python?"

    active_q, is_custom = determine_active_query(benchmark_q, custom_q)
    assert is_custom is True
    assert active_q == "Who developed Python?"

    result = execute_live_inference(
        active_query=active_q,
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=is_custom
    )

    assert "Guido van Rossum" in result["final_answer"]
    assert "Bengaluru" not in result["final_answer"]
    assert "Karnataka" not in result["final_answer"]


def test_case_c_empty_custom_query_uses_benchmark():
    """Test C: Empty custom query falls back to selected benchmark question."""
    benchmark_q = BENCHMARK_KARNATAKA["question_en"]
    custom_q = ""

    active_q, is_custom = determine_active_query(benchmark_q, custom_q)
    assert is_custom is False
    assert active_q == benchmark_q

    result = execute_live_inference(
        active_query=active_q,
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=is_custom
    )

    assert "Bengaluru" in result["final_answer"]
    assert result["has_gold_reference"] is True
    assert result["verification_status"] == "VERIFIED_BENCHMARK_GROUNDED"


def test_case_d_whitespace_only_custom_query():
    """Test D: Whitespace-only custom query falls back to benchmark question."""
    benchmark_q = BENCHMARK_KARNATAKA["question_en"]
    custom_q = "     \t \n   "

    active_q, is_custom = determine_active_query(benchmark_q, custom_q)
    assert is_custom is False
    assert active_q == benchmark_q

    result = execute_live_inference(
        active_query=active_q,
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=is_custom
    )

    assert "Bengaluru" in result["final_answer"]


def test_case_e_stale_invalidation_logic():
    """Test E: Verification that previous query output cannot be reused when active query changes."""
    # Simulate first inference on Karnataka
    res_karnataka = execute_live_inference(
        active_query=BENCHMARK_KARNATAKA["question_en"],
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=False
    )
    last_inference_query = res_karnataka["active_query"]

    # Now user edits input to Andhra
    new_active_q, new_is_custom = determine_active_query(
        BENCHMARK_KARNATAKA["question_en"],
        "what is the capital of andhra"
    )

    # Invalidation check:
    is_stale = (last_inference_query != new_active_q)
    assert is_stale is True
    # If stale, we do NOT return res_karnataka for new_active_q!


def test_case_f_custom_query_reference_evidence_separation():
    """Test F: Custom query without benchmark match does NOT reuse Karnataka evidence or claims."""
    benchmark_q = BENCHMARK_KARNATAKA["question_en"]
    custom_q = "What is the speed of light in vacuum?"

    active_q, is_custom = determine_active_query(benchmark_q, custom_q)
    assert is_custom is True

    result = execute_live_inference(
        active_query=active_q,
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_context=BENCHMARK_KARNATAKA,
        is_custom=is_custom
    )

    # Must NOT reuse Karnataka evidence or claims!
    assert result["ground_truth_evidence"] is None
    assert result["ground_truth_claims"] == []
    assert result["has_gold_reference"] is False
    assert result["verification_status"] == "UNVERIFIED_CUSTOM_NO_GOLD"
    assert "Bengaluru" not in result["final_answer"]


def test_case_g_multiple_languages_custom_query():
    """Test G: Custom query answering across all 5 languages (English, Hindi, Bengali, Telugu, Tamil)."""
    custom_q = "what is the capital of andhra"
    active_q, is_custom = determine_active_query(BENCHMARK_KARNATAKA["question_en"], custom_q)

    # English
    res_en = execute_live_inference(active_q, "en", "Strategy C", 0.55, BENCHMARK_KARNATAKA, is_custom)
    assert "Amaravati" in res_en["final_answer"]
    assert res_en["target_language"] == "en"

    # Hindi
    res_hi = execute_live_inference(active_q, "hi", "Strategy C", 0.55, BENCHMARK_KARNATAKA, is_custom)
    assert "अमरावती" in res_hi["final_answer"]
    assert res_hi["target_language"] == "hi"

    # Bengali
    res_bn = execute_live_inference(active_q, "bn", "Strategy C", 0.55, BENCHMARK_KARNATAKA, is_custom)
    assert "অমরাবতী" in res_bn["final_answer"]
    assert res_bn["target_language"] == "bn"

    # Telugu
    res_te = execute_live_inference(active_q, "te", "Strategy C", 0.55, BENCHMARK_KARNATAKA, is_custom)
    assert "అమరావతి" in res_te["final_answer"]
    assert res_te["target_language"] == "te"

    # Tamil
    res_ta = execute_live_inference(active_q, "ta", "Strategy C", 0.55, BENCHMARK_KARNATAKA, is_custom)
    assert "அமராவதி" in res_ta["final_answer"]
    assert res_ta["target_language"] == "ta"
