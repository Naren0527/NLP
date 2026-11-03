"""
Comprehensive Unit Tests for src/confidence.py.
Executes offline without downloading or loading Gemma or requiring CUDA.
"""

import math
import pytest
import torch

from src.confidence import (
    VerbalizedConfidenceParser,
    StatisticalConfidenceCalculator,
    ConfidenceEngine,
    normalize_unicode_digits,
    MetricSummary,
)


# =====================================================================
# 1. VERBALIZED CONFIDENCE PARSER TESTS (Strict NN% Percentage Contract)
# =====================================================================

class TestVerbalizedConfidenceParser:
    """Tests the strict regex parser r"^\\s*(\\d+(?:\\.\\d+)?)\\s*%\\s*$"."""

    @pytest.mark.parametrize("valid_input, expected_percent, expected_score", [
        ("85%", 85.0, 0.85),
        (" 85% ", 85.0, 0.85),
        ("85.5%", 85.5, 0.855),
        ("0%", 0.0, 0.0),
        ("1%", 1.0, 0.01),
        ("100%", 100.0, 1.0),
        ("100.0%", 100.0, 1.0),
        ("\n85%\n", 85.0, 0.85),
    ])
    def test_valid_percentages_accepted(self, valid_input, expected_percent, expected_score):
        res = VerbalizedConfidenceParser.parse(valid_input)
        assert res.is_valid is True
        assert res.raw_response == valid_input
        assert res.confidence_percent == expected_percent
        assert res.confidence_score == expected_score
        assert res.parsing_error is None

    @pytest.mark.parametrize("bare_input", [
        "85",
        " 85 ",
        "1",
        "1.0",
        "0.85",
        "0",
        "100",
    ])
    def test_bare_numbers_strictly_rejected(self, bare_input):
        res = VerbalizedConfidenceParser.parse(bare_input)
        assert res.is_valid is False
        assert res.confidence_score is None
        assert res.confidence_percent is None
        assert res.parsing_error == "BARE_NUMBER_MISSING_PERCENT_SYMBOL"

    @pytest.mark.parametrize("out_of_range_input", [
        "101%",
        "150%",
        "200%",
        "-5%",
    ])
    def test_out_of_range_rejected(self, out_of_range_input):
        res = VerbalizedConfidenceParser.parse(out_of_range_input)
        assert res.is_valid is False
        assert res.confidence_score is None
        assert "OUT_OF_RANGE" in str(res.parsing_error) or "MALFORMED" in str(res.parsing_error)

    @pytest.mark.parametrize("prose_input", [
        "I estimate 85%",
        "Confidence is 85%",
        "85% sure",
        "80% or 90%",
        "Between 70% and 80%",
        "True, 100%",
        "85% - based on knowledge",
    ])
    def test_prose_and_competing_percentages_rejected(self, prose_input):
        res = VerbalizedConfidenceParser.parse(prose_input)
        assert res.is_valid is False
        assert res.confidence_score is None
        assert res.parsing_error in ("PROSE_OR_EXPLANATION_PRESENT", "MULTIPLE_PERCENTAGES_FOUND")

    def test_empty_and_whitespace_rejected(self):
        res = VerbalizedConfidenceParser.parse("   ")
        assert res.is_valid is False
        assert res.parsing_error == "EMPTY_RESPONSE"

    def test_non_numeric_rejected(self):
        res = VerbalizedConfidenceParser.parse("I am completely certain of this answer.")
        assert res.is_valid is False
        assert res.parsing_error == "MALFORMED_NON_PERCENTAGE_RESPONSE"

    def test_unicode_indic_numerals_normalized_and_accepted(self):
        # Devanagari 85%
        devanagari_input = "८५%"
        res_hi = VerbalizedConfidenceParser.parse(devanagari_input)
        assert res_hi.is_valid is True
        assert res_hi.confidence_percent == 85.0
        assert res_hi.confidence_score == 0.85
        assert res_hi.raw_response == devanagari_input

        # Kannada ೮೫%
        kannada_input = "೮೫%"
        res_kn = VerbalizedConfidenceParser.parse(kannada_input)
        assert res_kn.is_valid is True
        assert res_kn.confidence_percent == 85.0

        # Tamil ௮௫%
        tamil_input = "௮௫%"
        res_ta = VerbalizedConfidenceParser.parse(tamil_input)
        assert res_ta.is_valid is True
        assert res_ta.confidence_percent == 85.0

        # Telugu ౮౫%
        telugu_input = "౮౫%"
        res_te = VerbalizedConfidenceParser.parse(telugu_input)
        assert res_te.is_valid is True
        assert res_te.confidence_percent == 85.0

        # Malayalam ൮൫%
        malayalam_input = "൮൫%"
        res_ml = VerbalizedConfidenceParser.parse(malayalam_input)
        assert res_ml.is_valid is True
        assert res_ml.confidence_percent == 85.0


# =====================================================================
# 2. STATISTICAL CONFIDENCE CALCULATOR TESTS (Hand-Calculated & Math)
# =====================================================================

class TestStatisticalConfidenceCalculator:
    """Verifies mathematical correctness, float32 precision, and special-token filtering."""

    def test_hand_calculated_analytical_example(self):
        """
        Manually construct a 3-step vocabulary distribution with exact known probabilities:
        Token 0 prob = 0.8  -> log_prob = ln(0.8)  ~ -0.22314355
        Token 1 prob = 0.5  -> log_prob = ln(0.5)  ~ -0.69314718
        Token 2 prob = 0.2  -> log_prob = ln(0.2)  ~ -1.60943791

        Expected Arithmetic Mean Prob = (0.8 + 0.5 + 0.2) / 3 = 1.5 / 3 = 0.500000
        Expected Arithmetic Mean Log Prob = (ln(0.8) + ln(0.5) + ln(0.2)) / 3 = ln(0.08) / 3 ~ -0.84190955
        Expected Geometric Mean Prob = (0.08)**(1/3) = exp(mean_log_prob) ~ 0.43088694
        """
        vocab_size = 10
        # Create unnormalized logits that produce these exact probabilities
        # Step 0: token 1 has prob 0.8, rest distributed equally
        logits_0 = torch.zeros(1, vocab_size)
        logits_0[0, 1] = 5.0
        # Normalize to exact target
        target_log_p0 = math.log(0.8)
        # We can construct synthetic logit tensors directly by inverting softmax
        step_logits = []
        for p, tok_id in [(0.8, 1), (0.5, 2), (0.2, 3)]:
            tensor = torch.full((1, vocab_size), -100.0)
            tensor[0, tok_id] = math.log(p)
            # Other tokens get remaining probability (1 - p) / (vocab_size - 1)
            other_p = (1.0 - p) / (vocab_size - 1)
            for j in range(vocab_size):
                if j != tok_id:
                    tensor[0, j] = math.log(other_p)
            step_logits.append(tensor)

        token_ids = [1, 2, 3]
        special_ids = set()

        res = StatisticalConfidenceCalculator.compute(
            step_logits=step_logits,
            generated_token_ids=token_ids,
            special_token_ids=special_ids,
        )

        assert res.is_valid is True
        summary = res.full_sequence
        assert summary.tokens_evaluated == 3

        # Verify against analytical manual values
        assert abs(summary.mean_token_probability - 0.500000) < 1e-4
        assert abs(summary.mean_log_probability - (-0.841910)) < 1e-4
        assert abs(summary.geometric_probability - 0.430887) < 1e-4

        # Verify mathematical identity: geom_mean == exp(mean_log_prob)
        assert abs(summary.geometric_probability - math.exp(summary.mean_log_probability)) < 1e-5

    def test_float32_precision_with_bfloat16_input(self):
        """Ensures bfloat16 tensors are cast to float32 so log-softmax does not suffer precision loss."""
        vocab_size = 50
        bf16_tensor = torch.randn(1, vocab_size, dtype=torch.bfloat16)
        token_ids = [5]
        res = StatisticalConfidenceCalculator.compute(
            step_logits=[bf16_tensor],
            generated_token_ids=token_ids,
            special_token_ids=set(),
        )
        assert res.is_valid is True
        assert isinstance(res.full_sequence.mean_token_probability, float)
        assert res.full_sequence.tokens_evaluated == 1

    def test_content_tokens_vs_full_sequence_special_token_filtering(self):
        """
        Sequence of 4 tokens: [101, 102, 103, 106].
        Token 106 is in special_token_ids (e.g. <end_of_turn>).
        Verify that:
        - full_sequence evaluates all 4 tokens (indices 0, 1, 2, 3)
        - content_tokens evaluates only 3 tokens (indices 0, 1, 2)
        - special_token_ids_found records [106]
        """
        vocab_size = 200
        step_logits = [torch.zeros(1, vocab_size) for _ in range(4)]
        token_ids = [101, 102, 103, 106]
        special_token_ids = {106, 1}

        res = StatisticalConfidenceCalculator.compute(
            step_logits=step_logits,
            generated_token_ids=token_ids,
            special_token_ids=special_token_ids
        )

        assert res.is_valid is True
        assert res.total_tokens_generated == 4
        assert res.special_token_count == 1
        assert res.special_token_ids_found == [106]

        # Content tokens check
        assert res.content_tokens.tokens_evaluated == 3
        assert res.content_tokens.token_indices == [0, 1, 2]

        # Full sequence check
        assert res.full_sequence.tokens_evaluated == 4
        assert res.full_sequence.token_indices == [0, 1, 2, 3]

    def test_empty_tokens_returns_invalid_without_exception(self):
        res = StatisticalConfidenceCalculator.compute(
            step_logits=[],
            generated_token_ids=[],
            special_token_ids=set(),
        )
        assert res.is_valid is False
        assert res.error_message == "EMPTY_TOKEN_SEQUENCE"
        assert res.content_tokens.tokens_evaluated == 0

    def test_out_of_bounds_token_id_raises_value_error(self):
        vocab_size = 50
        step_logits = [torch.zeros(1, vocab_size)]
        invalid_token_id = 999  # >= vocab_size
        with pytest.raises(ValueError) as excinfo:
            StatisticalConfidenceCalculator.compute(
                step_logits=step_logits,
                generated_token_ids=[invalid_token_id],
                special_token_ids=set(),
            )
        assert "out of vocabulary bounds" in str(excinfo.value)


# =====================================================================
# 3. CONFIDENCE ENGINE & PAIRING TESTS
# =====================================================================

class TestConfidenceEnginePairing:
    """Verifies that confidence prompt language matches answer language."""

    def test_prompt_language_matches_answer_language(self):
        engine = ConfidenceEngine(model_engine=None, prompts_path="configs/prompts.yaml")

        # Question in Kannada, Answer in Kannada -> Kannada confidence prompt
        p_kn, v_kn, l_kn = engine.get_confidence_prompt(
            question="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
            answer="ಬೆಂಗಳೂರು",
            answer_language="kn"
        )
        assert l_kn == "kn"
        assert v_kn == "v1.0_confidence_kn"
        assert "NN%" in p_kn
        assert "ವಿಶ್ವಾಸಾರ್ಹತೆ:" in p_kn

        # Question in Kannada, Answer in English -> English confidence prompt
        p_en, v_en, l_en = engine.get_confidence_prompt(
            question="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
            answer="The capital city of Karnataka is Bengaluru.",
            answer_language="en"
        )
        assert l_en == "en"
        assert v_en == "v1.0_confidence_en"
        assert "NN%" in p_en
        assert "Confidence:" in p_en

    def test_failed_verbalized_does_not_destroy_statistical_metrics(self):
        """Verifies that if verbalized confidence parsing fails, statistical metrics remain valid."""
        vocab_size = 100
        step_logits = [torch.zeros(1, vocab_size) for _ in range(3)]
        token_ids = [10, 20, 30]

        engine = ConfidenceEngine(model_engine=None)
        # Without model engine, verbalized confidence fails cleanly
        report = engine.extract_confidence_report(
            question="Test question?",
            answer_text="Test answer",
            answer_language="en",
            step_logits=step_logits,
            generated_token_ids=token_ids,
            special_token_ids=set()
        )

        assert report.statistical.is_valid is True
        assert report.statistical.content_tokens.tokens_evaluated == 3
        assert report.verbalized.is_valid is False
        assert report.verbalized.confidence_score is None
        assert report.extraction_status == "VERBALIZED_FAILED_STATISTICAL_OK"
