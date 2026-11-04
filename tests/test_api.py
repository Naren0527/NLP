"""
Unit tests for FastAPI endpoints and business logic in src/api.py.
Verifies health, model status, languages, benchmark summary, curated examples, and live inference.
"""

from src.api import (
    get_health,
    get_model_status,
    get_languages,
    get_benchmark_summary,
    get_benchmark_examples,
    run_inference,
    get_calibration_parameters,
    get_consistency_audit,
    get_strategies_benchmarks,
    get_reports_content,
    InferenceRequest,
)


def test_api_health():
    data = get_health()
    assert data["status"] == "online"
    assert "Gemma 3 4B" in data["model"]


def test_api_model_status():
    data = get_model_status()
    assert data["parameters"] == "4.3 Billion"
    assert data["inferences_recorded"] == 4050


def test_api_languages():
    langs = get_languages()
    assert "en" in langs and "hi" in langs and "te" in langs


def test_api_benchmark_summary():
    data = get_benchmark_summary()
    assert "partitions" in data
    assert "indic_aggregate" in data
    assert len(data["languages"]) == 5


def test_api_curated_examples():
    examples = get_benchmark_examples()
    assert len(examples) >= 5
    assert "Q_GEO_DIR_01" in [ex["id"] for ex in examples]


def test_api_inference_custom_query():
    req = InferenceRequest(
        query="what is the capital of andhra",
        benchmark_question="What is the capital city of Karnataka?",
        target_language="en",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_index=0
    )
    data = run_inference(req)
    assert data["is_custom"] is True
    assert "Amaravati" in data["final_answer"]
    assert "Bengaluru" not in data["final_answer"]
    assert "Karnataka" not in data["final_answer"]


def test_api_inference_fallback_to_benchmark():
    req = InferenceRequest(
        query="   ",
        benchmark_question="What is the capital city of Karnataka?",
        target_language="hi",
        strategy="Strategy C: Bilingual Verification & Arbitration (Winner)",
        tau_threshold=0.55,
        benchmark_index=0
    )
    data = run_inference(req)
    assert data["is_custom"] is False
    assert "बेंगलुरु" in data["final_answer"]


def test_api_calibration_parameters():
    data = get_calibration_parameters()
    assert "slope_w" in data["parameters"]
    assert "deltas" in data["parameters"]
    assert data["parameters"]["deltas"]["en"] == 0.0


def test_api_consistency_audit():
    data = get_consistency_audit()
    assert data["pairwise_comparisons_count"] == 750
    assert data["llm_judge_surface_agreement"] == 0.884
    assert len(data["error_taxonomy"]) == 3


def test_api_strategies_benchmarks():
    data = get_strategies_benchmarks()
    assert len(data["validation_strategy_selection"]) == 4
    assert len(data["strategy_e_frontier"]) == 10


def test_api_reports_content():
    docs = get_reports_content()
    assert "final_paper" in docs
    assert "methodology" in docs
