"""
Unit tests for src/answer_generator.py.
Runs offline using mocks without loading Gemma or requiring CUDA.
"""

from unittest.mock import MagicMock
import pytest

from src.answer_generator import AnswerGenerator, PairedAnswerResult
from src.model_engine import GenerationOutput


@pytest.fixture
def mock_model_engine():
    engine = MagicMock()
    
    def fake_generate(prompt: str, **kwargs):
        if "Confidence:" in prompt or "ವಿಶ್ವಾಸಾರ್ಹತೆ:" in prompt:
            text = "85%"
        elif "Answer the same question in English" in prompt:
            text = "English answer"
        else:
            text = "Original language answer"

        return GenerationOutput(
            prompt_text=prompt,
            formatted_prompt=f"<formatted>{prompt}</formatted>",
            answer_text=text,
            prompt_token_count=15,
            generated_token_count=8,
            total_token_count=23,
            generated_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            token_scores=[],
            token_logprobs=[-0.1] * 8,
            mean_token_probability=0.95,
            mean_log_probability=-0.1,
            geometric_probability=0.90,
            logits_type="unprocessed_logits (RAW)",
            execution_time_seconds=0.45,
            model_id="google/gemma-3-4b-it",
            model_revision="test_rev",
            device_info={"device": "cpu"},
            generation_config={"max_new_tokens": 128}
        )

    engine.generate.side_effect = fake_generate
    return engine


class TestAnswerGenerator:
    """Verifies dual independent generation, prompt formatting, and confidence elicitation."""

    def test_prompt_formatting(self):
        generator = AnswerGenerator(model_engine=MagicMock())
        orig_prompt, orig_v = generator.format_original_prompt("What is photosynthesis?", "Hindi")
        eng_prompt, eng_v = generator.format_english_prompt("What is photosynthesis?")

        assert "The user's language is Hindi" in orig_prompt
        assert "What is photosynthesis?" in orig_prompt
        assert orig_v == "v1.0_original"

        assert "Answer the same question in English" in eng_prompt
        assert "What is photosynthesis?" in eng_prompt
        assert eng_v == "v1.0_english"

    def test_paired_generation_without_confidence(self, mock_model_engine):
        generator = AnswerGenerator(model_engine=mock_model_engine)
        result = generator.generate_paired_answers(
            question="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
            language_code="kn",
            language_name="Kannada",
            elicit_confidence=False
        )

        assert isinstance(result, PairedAnswerResult)
        # Exactly 2 calls when confidence is disabled
        assert mock_model_engine.generate.call_count == 2
        assert result.original_answer.answer_text == "Original language answer"
        assert result.english_answer.answer_text == "English answer"
        assert result.original_confidence is None
        assert result.english_confidence is None

    def test_paired_generation_with_phase3_confidence(self, mock_model_engine):
        generator = AnswerGenerator(model_engine=mock_model_engine)
        result = generator.generate_paired_answers(
            question="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
            language_code="kn",
            language_name="Kannada",
            elicit_confidence=True
        )

        assert isinstance(result, PairedAnswerResult)
        # 4 calls: 2 answer generations + 2 localized confidence calls
        assert mock_model_engine.generate.call_count == 4

        # Original answer check
        assert result.original_answer.answer_text == "Original language answer"
        assert result.english_answer.answer_text == "English answer"

        # Confidence results check
        assert result.original_confidence is not None
        assert result.original_confidence.is_valid is True
        assert result.original_confidence.confidence_score == 0.85
        assert result.original_confidence.prompt_language == "kn"

        assert result.english_confidence is not None
        assert result.english_confidence.is_valid is True
        assert result.english_confidence.confidence_score == 0.85
        assert result.english_confidence.prompt_language == "en"
