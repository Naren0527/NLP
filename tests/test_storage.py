"""
Unit tests for src/storage.py.
Executes offline with temporary files without GPU or loading Gemma.
"""

import json
import os
import pytest

from src.storage import StorageEngine
from src.language_detector import DetectionResult
from src.model_engine import GenerationOutput
from src.answer_generator import PairedAnswerResult


@pytest.fixture
def sample_detection():
    return DetectionResult(
        raw_text="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
        language_code="kn",
        language_name="Kannada",
        script_name="Kannada",
        confidence=0.98,
        is_uncertain=False,
        requires_clarification=False,
        detection_method="deterministic_indic_script",
        is_overridden=False,
    )


@pytest.fixture
def sample_gen_output():
    return GenerationOutput(
        prompt_text="Sample prompt",
        formatted_prompt="<formatted>Sample prompt</formatted>",
        answer_text="ಬೆಂಗಳೂರು",
        prompt_token_count=12,
        generated_token_count=5,
        total_token_count=17,
        generated_token_ids=[101, 102, 103, 104, 105],
        token_scores=[],
        token_logprobs=[-0.05, -0.01, -0.02, -0.01, -0.04],
        mean_token_probability=0.975,
        mean_log_probability=-0.026,
        geometric_probability=0.974,
        logits_type="unprocessed_logits (RAW)",
        execution_time_seconds=0.35,
        model_id="google/gemma-3-4b-it",
        model_revision="commit_abc123",
        device_info={"device": "cuda:0", "allocated_mb": 3100.0},
        generation_config={"max_new_tokens": 128, "do_sample": False}
    )


class TestStorageEngine:
    """Verifies schema compliance, file persistence, and immutability."""

    def test_schema_completeness(self, tmp_path, sample_detection, sample_gen_output):
        output_file = str(tmp_path / "test_records.jsonl")
        storage = StorageEngine(output_path=output_file)

        record = storage.create_record(
            run_id="run_test_001",
            original_question=sample_detection.raw_text,
            detection_result=sample_detection,
            answer_language="kn",
            answer_stage="initial",
            prompt_version="v1.0_original",
            gen_output=sample_gen_output,
        )

        # Check required fields from blueprint schema
        expected_fields = [
            "run_id", "timestamp", "model_id", "model_revision", "quantization",
            "device_info", "software_versions", "original_question", "detected_language",
            "language_detection_confidence", "is_language_overridden", "answer_language",
            "answer_stage", "prompt_version", "exact_prompt", "formatted_prompt",
            "answer_text", "token_count", "prompt_token_count", "total_token_count",
            "generated_token_ids", "token_logprobs", "mean_token_probability",
            "mean_log_probability", "geometric_probability", "logits_type",
            "execution_time_seconds", "generation_configuration", "verbalized_confidence",
            "consistency_label", "correctness_label", "reference_answer", "verification_status"
        ]

        for field in expected_fields:
            assert field in record, f"Missing required schema field: {field}"

        assert record["run_id"] == "run_test_001"
        assert record["answer_language"] == "kn"
        assert record["token_count"] == 5
        assert len(record["generated_token_ids"]) == 5

    def test_append_immutability(self, tmp_path, sample_detection, sample_gen_output):
        output_file = str(tmp_path / "test_records.jsonl")
        storage = StorageEngine(output_path=output_file)

        # Create paired result
        paired = PairedAnswerResult(
            raw_question=sample_detection.raw_text,
            detected_language_code="kn",
            detected_language_name="Kannada",
            original_answer=sample_gen_output,
            english_answer=sample_gen_output,
            original_prompt_version="v1.0_original",
            english_prompt_version="v1.0_english",
        )

        # Save paired result (should create 2 records)
        rec1, rec2 = storage.save_paired_result(paired, sample_detection, run_id="run_pair_01")
        assert rec1["answer_language"] == "kn"
        assert rec2["answer_language"] == "en"

        # Verify file contents
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 2
        assert lines[0]["run_id"] == "run_pair_01"
        assert lines[1]["run_id"] == "run_pair_01"

        # Append another result without overwriting
        storage.save_paired_result(paired, sample_detection, run_id="run_pair_02")
        with open(output_file, "r", encoding="utf-8") as f:
            lines_after = [json.loads(l) for l in f if l.strip()]

        assert len(lines_after) == 4
        assert lines_after[0]["run_id"] == "run_pair_01"
        assert lines_after[2]["run_id"] == "run_pair_02"

    def test_failed_verbalized_confidence_preserves_valid_answer_record(
        self, tmp_path, sample_detection, sample_gen_output
    ):
        from src.confidence import VerbalizedConfidenceResult

        output_file = str(tmp_path / "test_failed_verb.jsonl")
        storage = StorageEngine(output_path=output_file)

        failed_verbalized = VerbalizedConfidenceResult(
            raw_response="Not sure about this.",
            confidence_score=None,
            confidence_percent=None,
            is_valid=False,
            parsing_error="MALFORMED_NON_PERCENTAGE_RESPONSE",
            prompt_version="v1.0_confidence_kn",
            elicitation_prompt="Sample prompt",
            prompt_language="kn",
            answer_language="kn",
            confidence_call_config={},
        )

        record = storage.create_record(
            run_id="run_failed_verb_01",
            original_question=sample_detection.raw_text,
            detection_result=sample_detection,
            answer_language="kn",
            answer_stage="initial",
            prompt_version="v1.0_original",
            gen_output=sample_gen_output,
            verbalized_conf=failed_verbalized,
        )

        # Core answer record must remain completely valid
        assert record["answer_text"] == "ಬೆಂಗಳೂರು"
        assert record["token_count"] == 5
        assert len(record["generated_token_ids"]) == 5
        assert record["mean_token_probability"] == 0.975

        # Verbalized confidence fields record the failure cleanly
        assert record["verbalized_confidence"] is None
        assert record["verbalized_confidence_valid"] is False
        assert record["verbalized_confidence_error"] == "MALFORMED_NON_PERCENTAGE_RESPONSE"
        assert record["confidence_extraction_status"] == "CONFIDENCE_UNEVALUATED" or "VERBALIZED_FAILED" in record["confidence_extraction_status"]
