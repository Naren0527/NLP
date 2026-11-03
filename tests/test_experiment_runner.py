"""
Unit tests for src/experiment_runner.py.
Executes offline with mocked ModelEngine without requiring CUDA.
"""

from unittest.mock import MagicMock
import os
import json
import pytest

from src.experiment_runner import ExperimentRunner
from src.model_engine import GenerationOutput


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.is_loaded.return_value = True
    
    def fake_generate(prompt: str, **kwargs):
        is_eng = "Answer the same question in English" in prompt
        return GenerationOutput(
            prompt_text=prompt,
            formatted_prompt=f"<formatted>{prompt}</formatted>",
            answer_text="English mock answer" if is_eng else "Kannada mock answer",
            prompt_token_count=10,
            generated_token_count=6,
            total_token_count=16,
            generated_token_ids=[10, 20, 30, 40, 50, 60],
            token_scores=[],
            token_logprobs=[-0.05] * 6,
            mean_token_probability=0.95,
            mean_log_probability=-0.05,
            geometric_probability=0.95,
            logits_type="unprocessed_logits (RAW)",
            execution_time_seconds=0.25,
            model_id="google/gemma-3-4b-it",
            model_revision="test_rev",
            device_info={"device": "cpu"},
            generation_config={"max_new_tokens": 128}
        )

    engine.generate.side_effect = fake_generate
    return engine


class TestExperimentRunner:
    """Verifies pipeline orchestration, uncertain states, and storage isolation."""

    def test_successful_run(self, tmp_path, mock_engine):
        out_file = str(tmp_path / "test_run.jsonl")
        runner = ExperimentRunner(model_engine=mock_engine, output_path=out_file)

        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"
        result = runner.run_pipeline(question=question)

        assert result["status"] == "SUCCESS"
        assert result["detection"].language_code == "kn"
        assert result["paired_result"] is not None

        # Check records written to JSONL
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 2
        assert lines[0]["answer_language"] == "kn"
        assert lines[1]["answer_language"] == "en"
        assert result["comparison_result"] is not None
        assert "consistency_label" in lines[0]
        assert "comparison_details" in lines[0]

    def test_uncertain_language_aborts_cleanly(self, tmp_path, mock_engine):
        out_file = str(tmp_path / "test_run_uncertain.jsonl")
        runner = ExperimentRunner(model_engine=mock_engine, output_path=out_file)

        # Ambiguous input (pure numbers)
        result = runner.run_pipeline(question="1234567")

        assert result["status"] == "UNCERTAIN_LANGUAGE"
        assert result["paired_result"] is None
        assert not os.path.exists(out_file)

    def test_manual_override_flow(self, tmp_path, mock_engine):
        out_file = str(tmp_path / "test_run_override.jsonl")
        runner = ExperimentRunner(model_engine=mock_engine, output_path=out_file)

        # Question with manual override to Hindi
        result = runner.run_pipeline(question="What is this?", override_lang="hi")

        assert result["status"] == "SUCCESS"
        assert result["detection"].language_code == "hi"
        assert result["detection"].is_overridden is True
