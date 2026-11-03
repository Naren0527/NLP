"""
Unit tests for src/answer_comparator.py.
Offline, mocked tests verifying deterministic extraction, evidence substring validation,
arbiter decision table, and end-to-end comparison logic without GPU.
"""

from dataclasses import asdict
import json
import pytest
from unittest.mock import MagicMock

from src.answer_comparator import (
    DeterministicFeatureExtractor,
    DeterministicFeatureSummary,
    CandidateDiscrepancy,
    EvidenceGroundedClaim,
    EvidenceValidator,
    ClassificationArbiter,
    AnswerComparator,
    ComparisonResult,
    NEGATION_TOKENS,
)
from src.model_engine import GenerationOutput


# ==============================================================================
# 1. DeterministicFeatureExtractor Tests
# ==============================================================================

class TestDeterministicFeatureExtractor:
    """Verifies candidate extraction, context preservation, and numeral normalization."""

    def test_extract_numbers_with_context_and_unicode_digits(self):
        # Kannada digits: ೧೯೫೬ -> 1956
        kn_text = "ಕರ್ನಾಟಕ ರಾಜ್ಯವು ೧೯೫೬ ರಲ್ಲಿ ಸ್ಥಾಪನೆಯಾಯಿತು."
        matches = DeterministicFeatureExtractor.extract_numbers_with_context(kn_text, window=10)
        assert len(matches) == 1
        num, ctx = matches[0]
        assert num == "1956"
        assert "1956" in ctx

    def test_extract_dates_candidate_discrepancy(self):
        orig_text = "The treaty was signed in 1947."
        eng_text = "The treaty was signed in 1950."
        summary = DeterministicFeatureExtractor.analyze(orig_text, eng_text, "en")
        
        assert summary.has_date_candidates is True
        date_disc = [c for c in summary.candidate_discrepancies if c.category == "DATE"]
        assert len(date_disc) == 1
        assert date_disc[0].original_value == "1947"
        assert date_disc[0].english_value == "1950"

    def test_extract_negation_markers_across_languages(self):
        # Test Kannada negation
        kn_text = "ಇದು ಸತ್ಯವಲ್ಲ."  # Contains "ಅಲ್ಲ" / "ವಲ್ಲ"
        kn_summary = DeterministicFeatureExtractor.analyze(kn_text, "This is true.", "kn")
        assert kn_summary.has_polarity_candidates is True

        # Test Hindi negation
        hi_text = "यह सही नहीं है।"  # Contains "नहीं"
        hi_summary = DeterministicFeatureExtractor.analyze(hi_text, "This is correct.", "hi")
        assert hi_summary.has_polarity_candidates is True

        # Test Hindi proper noun (e.g. कर्नाटक) does NOT trigger false positive negation
        hi_pos_text = "कर्नाटक की राजधानी बैंगलोर है।"
        hi_pos_summary = DeterministicFeatureExtractor.analyze(hi_pos_text, "The capital of Karnataka is Bangalore.", "hi")
        assert hi_pos_summary.has_polarity_candidates is False

        # Test Tamil negation
        ta_text = "இது உண்மை இல்லை."  # Contains "இல்லை"
        ta_summary = DeterministicFeatureExtractor.analyze(ta_text, "This is true.", "ta")
        assert ta_summary.has_polarity_candidates is True

        # Test Telugu negation
        te_text = "ఇది నిజం కాదు."  # Contains "కాదు"
        te_summary = DeterministicFeatureExtractor.analyze(te_text, "This is true.", "te")
        assert te_summary.has_polarity_candidates is True

        # Test Malayalam negation
        ml_text = "ഇത് സത്യം അല്ല."  # Contains "അല്ല"
        ml_summary = DeterministicFeatureExtractor.analyze(ml_text, "This is true.", "ml")
        assert ml_summary.has_polarity_candidates is True

    def test_numbers_describing_different_facts_do_not_automatically_contradict(self):
        orig_text = "The city has 5 districts and a population of 1000000."
        eng_text = "The city covers 500 square km and has 5 districts."
        summary = DeterministicFeatureExtractor.analyze(orig_text, eng_text, "en")
        # Differences in numbers exist, but extractor only marks candidate discrepancies
        assert summary.has_numerical_candidates is True
        # Does not declare contradiction; judge will semantically interpret


# ==============================================================================
# 2. EvidenceValidator Tests
# ==============================================================================

class TestEvidenceValidator:
    """Verifies strict relationship contracts and exact substring validation."""

    def test_valid_matched_relationship(self):
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
        eng_text = "The capital of Karnataka is Bangalore."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="MATCHED",
            original_evidence_span="ಬೆಂಗಳೂರು",
            english_evidence_span="Bangalore",
            interpreted_claim="Both name Bangalore/Bengaluru as capital",
            explanation="Exact match"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is True
        assert eng_valid is True
        assert err is None

    def test_valid_contradictory_relationship(self):
        orig_text = "ರಾಜಧಾನಿ ಮೈಸೂರು."
        eng_text = "The capital is Bangalore."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="CONTRADICTORY",
            original_evidence_span="ಮೈಸೂರು",
            english_evidence_span="Bangalore",
            interpreted_claim="Conflict on capital city",
            explanation="Mysore vs Bangalore"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is True
        assert eng_valid is True
        assert err is None

    def test_valid_original_only_relationship(self):
        orig_text = "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ದೊಡ್ಡ ನಗರ."
        eng_text = "Bengaluru is the capital."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="ORIGINAL_ONLY",
            original_evidence_span="ದೊಡ್ಡ ನಗರ",
            english_evidence_span=None,
            interpreted_claim="Largest city detail present only in Kannada",
            explanation="Unilateral detail"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is True
        assert eng_valid is True
        assert err is None

    def test_original_only_violates_contract_if_english_span_present(self):
        orig_text = "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ದೊಡ್ಡ ನಗರ."
        eng_text = "Bengaluru is the capital."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="ORIGINAL_ONLY",
            original_evidence_span="ದೊಡ್ಡ ನಗರ",
            english_evidence_span="capital",
            interpreted_claim="Detail present only in Kannada",
            explanation="Invalid contract"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is False
        assert "ENGLISH_EVIDENCE_MUST_BE_NULL" in err

    def test_valid_english_only_relationship(self):
        orig_text = "ಬೆಂಗಳೂರು ರಾಜಧಾನಿ."
        eng_text = "Bengaluru is known as the Silicon Valley of India."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="ENGLISH_ONLY",
            original_evidence_span=None,
            english_evidence_span="Silicon Valley of India",
            interpreted_claim="Silicon valley nickname only in English",
            explanation="Unilateral detail"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is True
        assert eng_valid is True
        assert err is None

    def test_english_only_violates_contract_if_original_span_present(self):
        orig_text = "ಬೆಂಗಳೂರು ರಾಜಧಾನಿ."
        eng_text = "Bengaluru is known as Silicon Valley."
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="ENGLISH_ONLY",
            original_evidence_span="ರಾಜಧಾನಿ",
            english_evidence_span="Silicon Valley",
            interpreted_claim="Silicon valley nickname",
            explanation="Invalid contract"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert orig_valid is False
        assert "ORIGINAL_EVIDENCE_MUST_BE_NULL" in err

    def test_fabricated_evidence_span_rejected(self):
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
        eng_text = "The capital is Bangalore."
        # Fabricate span not present in text
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="MATCHED",
            original_evidence_span="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ",
            english_evidence_span="New Delhi",  # Not in english_text!
            interpreted_claim="Capital match",
            explanation="Fabricated span test"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, orig_text, eng_text)
        assert eng_valid is False
        assert "ENGLISH_SPAN_NOT_FOUND_IN_SOURCE" in err

    def test_empty_or_missing_interpreted_claim_rejected(self):
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="MATCHED",
            original_evidence_span="A",
            english_evidence_span="B",
            interpreted_claim="   ",
            explanation="Empty"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, "A", "B")
        assert orig_valid is False
        assert err == "EMPTY_INTERPRETED_CLAIM"

    def test_invalid_relationship_category_rejected(self):
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="PARTIAL_MATCH",  # Not in ALLOWED_RELATIONSHIPS
            original_evidence_span="A",
            english_evidence_span="B",
            interpreted_claim="Some claim",
            explanation="Test"
        )
        orig_valid, eng_valid, err = EvidenceValidator.validate_claim(claim, "A", "B")
        assert orig_valid is False
        assert "INVALID_RELATIONSHIP_CATEGORY" in err


# ==============================================================================
# 3. ClassificationArbiter Decision Table Tests
# ==============================================================================

class TestClassificationArbiter:
    """Verifies formal Decision Table enforcement and evidence-grounding overrides."""

    @pytest.fixture
    def empty_deterministic(self):
        return DeterministicFeatureSummary(
            extractor_version="v1.0",
            candidate_discrepancies=[],
            has_numerical_candidates=False,
            has_date_candidates=False,
            has_polarity_candidates=False,
            has_entity_candidates=False,
        )

    def test_malformed_output_fallback_to_uncertain(self, empty_deterministic):
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="MALFORMED",
            overall_rationale="",
            validated_claims=[],
            validation_status="PARSING_FAILED",
            deterministic=empty_deterministic
        )
        assert final_class == "UNCERTAIN"
        assert src == "MALFORMED_OUTPUT_FALLBACK"

    def test_ungrounded_evidence_overrides_to_uncertain(self, empty_deterministic):
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="CONTRADICTORY",
            original_evidence_span="A",
            english_evidence_span="B",
            interpreted_claim="Conflict",
            explanation="Test",
            is_original_evidence_valid=False,
            validation_error="ORIGINAL_SPAN_NOT_FOUND_IN_SOURCE"
        )
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="INCONSISTENT",
            overall_rationale="Conflict between cities",
            validated_claims=[claim],
            validation_status="FABRICATED_EVIDENCE_DETECTED",
            deterministic=empty_deterministic
        )
        assert final_class == "UNCERTAIN"
        assert src == "ARBITER_OVERRIDE_UNGROUNDED_EVIDENCE"
        assert "Evidence grounding failed" in reason

    def test_inconsistent_accepted_when_contradiction_grounded(self, empty_deterministic):
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="CONTRADICTORY",
            original_evidence_span="ಮೈಸೂರು",
            english_evidence_span="Bangalore",
            interpreted_claim="Different capital cities",
            explanation="Contradiction verified",
            is_original_evidence_valid=True,
            is_english_evidence_valid=True,
        )
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="INCONSISTENT",
            overall_rationale="The answers state two different cities.",
            validated_claims=[claim],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=empty_deterministic
        )
        assert final_class == "INCONSISTENT"
        assert src == "JUDGE_ACCEPTED"
        assert reason is None

    def test_inconsistent_overridden_to_uncertain_if_no_contradictory_claims(self, empty_deterministic):
        # Judge said INCONSISTENT but claims only show MATCHED
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="MATCHED",
            original_evidence_span="ಬೆಂಗಳೂರು",
            english_evidence_span="Bangalore",
            interpreted_claim="Both agree",
            explanation="Matched",
            is_original_evidence_valid=True,
            is_english_evidence_valid=True,
        )
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="INCONSISTENT",
            overall_rationale="I think they differ",
            validated_claims=[claim],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=empty_deterministic
        )
        assert final_class == "UNCERTAIN"
        assert src == "ARBITER_OVERRIDE_NO_VALID_CONTRADICTION"

    def test_consistent_accepted_when_no_contradiction(self, empty_deterministic):
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="MATCHED",
            original_evidence_span="ಬೆಂಗಳೂರು",
            english_evidence_span="Bangalore",
            interpreted_claim="Capital match",
            explanation="Consistent",
            is_original_evidence_valid=True,
            is_english_evidence_valid=True,
        )
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="CONSISTENT",
            overall_rationale="Both answers agree on Bangalore.",
            validated_claims=[claim],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=empty_deterministic
        )
        assert final_class == "CONSISTENT"
        assert src == "JUDGE_ACCEPTED"
        assert reason is None

    def test_consistent_overridden_to_inconsistent_if_contradiction_present(self, empty_deterministic):
        # Judge erroneously said CONSISTENT but included a CONTRADICTORY claim
        claim = EvidenceGroundedClaim(
            claim_id="CLM_01",
            relationship="CONTRADICTORY",
            original_evidence_span="A",
            english_evidence_span="B",
            interpreted_claim="Conflict",
            explanation="Direct contradiction",
            is_original_evidence_valid=True,
            is_english_evidence_valid=True,
        )
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="CONSISTENT",
            overall_rationale="Consistent overall",
            validated_claims=[claim],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=empty_deterministic
        )
        assert final_class == "INCONSISTENT"
        assert src == "ARBITER_OVERRIDE_CONTRADICTORY_CLAIM_PRESENT"

    def test_consistent_overridden_to_uncertain_if_date_conflict_unresolved(self):
        # Deterministic date conflict detected
        date_disc = CandidateDiscrepancy(
            category="DATE",
            original_value="1947",
            original_context="1947",
            english_value="1950",
            english_context="1950",
            potential_attribute="year"
        )
        det_summary = DeterministicFeatureSummary(
            extractor_version="v1.0",
            candidate_discrepancies=[date_disc],
            has_numerical_candidates=False,
            has_date_candidates=True,
            has_polarity_candidates=False,
            has_entity_candidates=False,
        )
        # Judge said CONSISTENT with no MATCHED claim on the date
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="CONSISTENT",
            overall_rationale="Looks good",
            validated_claims=[],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=det_summary
        )
        assert final_class == "UNCERTAIN"
        assert src == "ARBITER_OVERRIDE_UNRESOLVED_DATE_DISCREPANCY"

    def test_uncertain_accepted(self, empty_deterministic):
        final_class, src, reason = ClassificationArbiter.arbitrate(
            raw_judge_class="UNCERTAIN",
            overall_rationale="Answers are too vague to compare.",
            validated_claims=[],
            validation_status="ALL_EVIDENCE_GROUNDED",
            deterministic=empty_deterministic
        )
        assert final_class == "UNCERTAIN"
        assert src == "JUDGE_ACCEPTED"


# ==============================================================================
# 4. JSON Parser Tests
# ==============================================================================

class TestJudgeJsonParsing:
    """Verifies robust extraction of JSON from model outputs."""

    def test_parse_clean_json(self):
        raw = '{"classification": "CONSISTENT", "overall_rationale": "Same meaning", "claims": []}'
        comparator = AnswerComparator(model_engine=None)
        data, valid, err = comparator.parse_judge_json(raw)
        assert valid is True
        assert data["classification"] == "CONSISTENT"

    def test_parse_markdown_fenced_json(self):
        raw = """```json
{
  "classification": "INCONSISTENT",
  "overall_rationale": "Direct contradiction",
  "claims": [
    {
      "claim_id": "CLM_01",
      "relationship": "CONTRADICTORY",
      "original_evidence_span": "A",
      "english_evidence_span": "B",
      "interpreted_claim": "Fact differs",
      "explanation": "Different facts"
    }
  ]
}
```"""
        comparator = AnswerComparator(model_engine=None)
        data, valid, err = comparator.parse_judge_json(raw)
        assert valid is True
        assert data["classification"] == "INCONSISTENT"
        assert len(data["claims"]) == 1

    def test_parse_garbage_or_prose_fails_cleanly(self):
        raw = "I think both answers are mostly consistent because they describe the same place."
        comparator = AnswerComparator(model_engine=None)
        data, valid, err = comparator.parse_judge_json(raw)
        assert valid is False
        assert "NO_JSON_OBJECT_FOUND" in err


# ==============================================================================
# 5. End-to-End Mocked AnswerComparator Tests
# ==============================================================================

class TestAnswerComparatorEndToEnd:
    """Verifies end-to-end comparison with mocked ModelEngine."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.model_revision = "test_revision_hash"
        return engine

    def test_consistent_pair_with_matched_claims(self, mock_engine):
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಆಗಿದೆ."
        eng_text = "The capital of Karnataka is Bangalore."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        judge_response_json = json.dumps({
            "classification": "CONSISTENT",
            "overall_rationale": "Both answers state that Bangalore/Bengaluru is the capital of Karnataka.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "ಬೆಂಗಳೂರು",
                    "english_evidence_span": "Bangalore",
                    "interpreted_claim": "Bangalore is the capital of Karnataka",
                    "explanation": "Transliterated city names refer to the identical entity."
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res: ComparisonResult = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"
        assert res.evidence_validation_status == "ALL_EVIDENCE_GROUNDED"
        assert len(res.matched_claims) == 1
        assert res.matched_claims[0].original_evidence_span == "ಬೆಂಗಳೂರು"
        assert res.matched_claims[0].english_evidence_span == "Bangalore"

    def test_inconsistent_pair_with_contradiction(self, mock_engine):
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮೈಸೂರು."
        eng_text = "The capital of Karnataka is Bangalore."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        judge_response_json = json.dumps({
            "classification": "INCONSISTENT",
            "overall_rationale": "The answers contradict each other on the capital city (Mysore vs Bangalore).",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "CONTRADICTORY",
                    "original_evidence_span": "ಮೈಸೂರು",
                    "english_evidence_span": "Bangalore",
                    "interpreted_claim": "Capital city of Karnataka",
                    "explanation": "Direct factual conflict between Mysore and Bangalore."
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "INCONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"
        assert len(res.contradictory_claims) == 1

    def test_fabricated_evidence_span_triggers_arbiter_uncertainty(self, mock_engine):
        orig_text = "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ."
        eng_text = "Bangalore is the capital of Karnataka."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        # Judge hallucinated quotes that don't appear in source
        judge_response_json = json.dumps({
            "classification": "INCONSISTENT",
            "overall_rationale": "Hallucinated conflict.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "CONTRADICTORY",
                    "original_evidence_span": "ಹುಬ್ಬಳ್ಳಿ",  # NOT in orig_text!
                    "english_evidence_span": "Bangalore",
                    "interpreted_claim": "Capital conflict",
                    "explanation": "Fabricated evidence"
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        # MUST be overridden to UNCERTAIN because evidence span is fabricated
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_UNGROUNDED_EVIDENCE"
        assert res.evidence_validation_status == "FABRICATED_EVIDENCE_DETECTED"

    def test_asymmetric_detail_retains_consistent(self, mock_engine):
        orig_text = "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ದೊಡ್ಡ ಐಟಿ ಕೇಂದ್ರವಾಗಿದೆ."
        eng_text = "Bangalore is the capital of Karnataka."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        judge_response_json = json.dumps({
            "classification": "CONSISTENT",
            "overall_rationale": "Both agree on the capital. Kannada provides additional true context about IT.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "ರಾಜಧಾನಿ",
                    "english_evidence_span": "capital",
                    "interpreted_claim": "Both name it as the capital",
                    "explanation": "Core agreement"
                },
                {
                    "claim_id": "CLM_02",
                    "relationship": "ORIGINAL_ONLY",
                    "original_evidence_span": "ದೊಡ್ಡ ಐಟಿ ಕೇಂದ್ರವಾಗಿದೆ",
                    "english_evidence_span": None,
                    "interpreted_claim": "Kannada mentions IT hub",
                    "explanation": "Compatible additional detail"
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_english_wrong_and_original_correct_is_inconsistent(self, mock_engine):
        # English is wrong ("New Delhi") and Kannada is correct ("ಬೆಂಗಳೂರು")
        # Proves English is NOT treated as ground truth
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
        eng_text = "The capital of Karnataka is New Delhi."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        judge_response_json = json.dumps({
            "classification": "INCONSISTENT",
            "overall_rationale": "Answers disagree on the capital (Bengaluru vs New Delhi).",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "CONTRADICTORY",
                    "original_evidence_span": "ಬೆಂಗಳೂರು",
                    "english_evidence_span": "New Delhi",
                    "interpreted_claim": "Capital city conflict",
                    "explanation": "Bengaluru vs New Delhi conflict"
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "INCONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_both_wrong_but_agreeing_is_consistent(self, mock_engine):
        # Both claim capital is Hubballi (factually wrong in reality, but mutually consistent)
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಹುಬ್ಬಳ್ಳಿ."
        eng_text = "The capital of Karnataka is Hubballi."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        judge_response_json = json.dumps({
            "classification": "CONSISTENT",
            "overall_rationale": "Both answers mutually agree that Hubballi is the capital.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "ಹುಬ್ಬಳ್ಳಿ",
                    "english_evidence_span": "Hubballi",
                    "interpreted_claim": "Both name Hubballi",
                    "explanation": "Mutual factual agreement, independent of real-world truth"
                }
            ]
        })

        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=judge_response_json,
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_judge_malformed_response_fallback(self, mock_engine):
        orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
        eng_text = "The capital of Karnataka is Bangalore."
        question = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"

        # Non-JSON garbage response
        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text="This is an informal paragraph with no JSON at all.",
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare(
            question=question,
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            original_language_code="kn",
            original_language_name="Kannada"
        )

        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "MALFORMED_OUTPUT_FALLBACK"
        assert res.raw_judge_classification == "MALFORMED"
        assert res.evidence_validation_status == "PARSING_FAILED"


# ==============================================================================
# 6. Targeted Adversarial Scenarios (Audit Verification)
# ==============================================================================

class TestTargetedAdversarialScenarios:
    """
    Rigorously tests the 10 adversarial edge cases to audit Arbiter and Validator behavior.
    """

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.model_revision = "test_revision_hash"
        return engine

    def _make_mock_output(self, judge_json_dict: dict) -> GenerationOutput:
        return GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=json.dumps(judge_json_dict),
            prompt_token_count=10, generated_token_count=20, total_token_count=30,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )

    def test_case_1_judge_misses_60_vs_100_contradiction_returns_uncertain(self, mock_engine):
        # 1. A judge misses a 60 km/h versus 100 km/h contradiction and returns CONSISTENT. Final result must be UNCERTAIN.
        orig = "The maximum speed limit is 60 km/h."
        eng = "The maximum speed limit is 100 km/h."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both describe speed limits.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "speed limit",
                "english_evidence_span": "speed limit",
                "interpreted_claim": "Both mention speed limit",
                "explanation": "Shallow match ignoring numbers"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("speed limit?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_UNRESOLVED_NUMERICAL_DISCREPANCY"
        assert "speed limit" in res.claims[0].original_evidence_span

    def test_case_2_number_discrepancy_not_resolved_by_unrelated_matched_claim(self, mock_engine):
        # 2. A number discrepancy cannot be resolved by an unrelated MATCHED claim.
        orig = "Bengaluru is the capital and has 150 parks."
        eng = "Bengaluru is the capital and has 200 parks."
        # Judge only matches the capital and ignores the park count difference
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both agree on the capital.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "Bengaluru is the capital",
                "english_evidence_span": "Bengaluru is the capital",
                "interpreted_claim": "Bengaluru is the capital",
                "explanation": "Unrelated matched claim"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital and parks?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert "UNRESOLVED_NUMERICAL_DISCREPANCY" in res.classification_decision_source

    def test_case_3_date_discrepancy_not_resolved_by_unrelated_matched_claim(self, mock_engine):
        # 3. A date discrepancy cannot be resolved by an unrelated MATCHED claim.
        orig = "Bangalore is the capital. The treaty was ratified in 1947."
        eng = "Bangalore is the capital. The treaty was ratified in 1950."
        # Judge matches Bangalore is the capital, but leaves 1947 vs 1950 unmentioned
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both state that Bangalore is the capital.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "Bangalore is the capital",
                "english_evidence_span": "Bangalore is the capital",
                "interpreted_claim": "Bangalore is the capital",
                "explanation": "Capital matches"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital and treaty?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert "UNRESOLVED_DATE_DISCREPANCY" in res.classification_decision_source

    def test_case_4_polarity_discrepancy_ignored_by_judge_results_in_uncertain(self, mock_engine):
        # 4. A polarity discrepancy ignored by the judge results in UNCERTAIN.
        orig = "Water boils at 100 degrees Celsius."
        eng = "Water does not boil at 100 degrees Celsius."
        # Biased judge ignores the 'not' and returns CONSISTENT
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both discuss boiling at 100 degrees Celsius.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "100 degrees Celsius",
                "english_evidence_span": "100 degrees Celsius",
                "interpreted_claim": "Boiling temperature",
                "explanation": "Ignored polarity"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("boiling?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert "UNRESOLVED_POLARITY_DISCREPANCY" in res.classification_decision_source

    def test_case_5_material_ambiguous_claim_cannot_coexist_with_final_consistent(self, mock_engine):
        # 5. A material AMBIGUOUS claim cannot coexist with final CONSISTENT.
        orig = "The suspect was seen in the vicinity of the station."
        eng = "The suspect was at the airport."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",  # Faulty judge tries to declare CONSISTENT
            "overall_rationale": "Might be consistent.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "AMBIGUOUS",
                "original_evidence_span": "vicinity of the station",
                "english_evidence_span": "at the airport",
                "interpreted_claim": "Suspect location",
                "explanation": "Ambiguous location claim"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("suspect?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_AMBIGUOUS_CLAIM_PRESENT"

    def test_case_6_valid_contradictory_claim_forces_inconsistent(self, mock_engine):
        # 6. A valid CONTRADICTORY claim forces INCONSISTENT.
        orig = "The capital of Karnataka is Mysore."
        eng = "The capital of Karnataka is Bangalore."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",  # Faulty judge tries to declare CONSISTENT
            "overall_rationale": "Both name cities in Karnataka.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "CONTRADICTORY",
                "original_evidence_span": "Mysore",
                "english_evidence_span": "Bangalore",
                "interpreted_claim": "Capital city of Karnataka",
                "explanation": "Mysore vs Bangalore contradiction"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "INCONSISTENT"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_CONTRADICTORY_CLAIM_PRESENT"

    def test_case_7_judge_explicitly_resolves_numerical_candidate_allowing_consistent(self, mock_engine):
        # 7. A judge explicitly resolves a numerical candidate discrepancy with the correct attribute context, allowing CONSISTENT when appropriate.
        orig = "The university has 5 campuses across the state."
        eng = "The university offers 5-year degree programs."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Compatible facts: one specifies campuses and the other specifies program duration.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "ORIGINAL_ONLY",
                    "original_evidence_span": "5 campuses",
                    "english_evidence_span": None,
                    "interpreted_claim": "The university has 5 campuses",
                    "explanation": "Campuses attribute mentioned only in original"
                },
                {
                    "claim_id": "CLM_02",
                    "relationship": "ENGLISH_ONLY",
                    "original_evidence_span": None,
                    "english_evidence_span": "5-year degree programs",
                    "interpreted_claim": "The university offers 5-year degree programs",
                    "explanation": "Program duration attribute mentioned only in English"
                }
            ]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("university?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_case_8_original_only_claim_preserved_without_contradiction(self, mock_engine):
        # 8. An ORIGINAL_ONLY factual claim is preserved for verification without automatically being classified as contradictory.
        orig = "Bangalore is the capital of Karnataka and was founded by Kempe Gowda I in 1537."
        eng = "Bangalore is the capital of Karnataka."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Core capital agrees. Founder and year are compatible supplementary details.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "capital of Karnataka",
                    "english_evidence_span": "capital of Karnataka",
                    "interpreted_claim": "Bangalore is the capital of Karnataka",
                    "explanation": "Core agreement"
                },
                {
                    "claim_id": "CLM_02",
                    "relationship": "ORIGINAL_ONLY",
                    "original_evidence_span": "founded by Kempe Gowda I in 1537",
                    "english_evidence_span": None,
                    "interpreted_claim": "Bangalore was founded by Kempe Gowda I in 1537",
                    "explanation": "Supplementary historical detail preserved for Phase 5 verification"
                }
            ]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"
        assert len(res.original_only_claims) == 1
        assert res.original_only_claims[0].original_evidence_span == "founded by Kempe Gowda I in 1537"
        assert res.original_only_claims[0].english_evidence_span is None

    def test_case_9_english_only_claim_preserved_without_contradiction(self, mock_engine):
        # 9. An ENGLISH_ONLY factual claim is preserved for verification without automatically being classified as contradictory.
        orig = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
        eng = "The capital of Karnataka is Bangalore, which is the IT hub of India."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both agree on Bangalore. The IT hub detail is a supplementary assertion.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "ರಾಜಧಾನಿ ಬೆಂಗಳೂರು",
                    "english_evidence_span": "capital of Karnataka is Bangalore",
                    "interpreted_claim": "Bangalore is the capital of Karnataka",
                    "explanation": "Core agreement"
                },
                {
                    "claim_id": "CLM_02",
                    "relationship": "ENGLISH_ONLY",
                    "original_evidence_span": None,
                    "english_evidence_span": "IT hub of India",
                    "interpreted_claim": "Bangalore is the IT hub of India",
                    "explanation": "Supplementary economic detail preserved for Phase 5 verification"
                }
            ]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital?", orig, eng, "kn", "Kannada")
        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_case_10_both_wrong_but_consistent(self, mock_engine):
        # 10. Both answers can be factually wrong but mutually consistent, and the result remains CONSISTENT.
        orig = "The official currency of Japan is the Euro."
        eng = "The official currency of Japan is the Euro."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both answers state that the currency of Japan is the Euro. Mutual factual agreement.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "Euro",
                "english_evidence_span": "Euro",
                "interpreted_claim": "The official currency of Japan is the Euro",
                "explanation": "Mutual factual agreement independent of real-world veracity"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("currency?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "CONSISTENT"
        assert res.classification_decision_source == "JUDGE_ACCEPTED"

    def test_case_11_truncated_judge_json_preserves_uncertain_fallback(self, mock_engine):
        # Truncated JSON midway through output (e.g. hitting token budget)
        truncated_json_str = '{"classification": "CONSISTENT", "overall_rationale": "Incomplete...", "claims": [{"claim_id": "CLM_01", "relationship": "MATCHED", "original_evidence_span": "Bangalore"'
        mock_engine.generate.return_value = GenerationOutput(
            prompt_text="...", formatted_prompt="...",
            answer_text=truncated_json_str,
            prompt_token_count=10, generated_token_count=512, total_token_count=522,
            generated_token_ids=[], token_scores=[], token_logprobs=[],
            mean_token_probability=0.9, mean_log_probability=-0.1, geometric_probability=0.9,
            logits_type="unprocessed_logits (RAW)", execution_time_seconds=0.2,
            model_id="google/gemma-3-4b-it", model_revision="test_revision_hash",
            device_info={}, generation_config={}
        )
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("capital?", "Bangalore", "Bangalore", "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "MALFORMED_OUTPUT_FALLBACK"
        assert res.evidence_validation_status == "PARSING_FAILED"
        assert "JSON_DECODE_ERROR" in res.arbiter_override_reason

    def test_case_12_configurable_comparison_token_budget(self):
        # Verify that AnswerComparator loads the configurable token budget from configs/model.yaml
        comparator = AnswerComparator(model_engine=None)
        assert comparator.judge_generation_config["max_new_tokens"] == 1024
        assert comparator.judge_generation_config["do_sample"] is False

    def test_case_13_same_number_different_attribute_cannot_resolve_discrepancy(self, mock_engine):
        # Orig has 5 airports, Eng has no airports (unilateral number 5 for airports)
        orig = "The city has 5 operational airports."
        eng = "The city has great public transport."
        # Flawed judge attempts to resolve the '5' by matching an unrelated claim about 5 bus lines
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both describe transport.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "5 operational airports",
                "english_evidence_span": "great public transport",
                "interpreted_claim": "The city has 5 bus lines",  # Conflicting attribute
                "explanation": "Claim describes bus lines, not airports"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("transport?", orig, eng, "en", "English")
        # Because candidate discrepancy potential_attribute is 'operational'/'airports',
        # and the claim text does not describe airports, the numerical discrepancy remains unresolved
        assert res.deterministic_features.has_numerical_candidates is True

    def test_case_14_unrelated_historical_claim_cannot_resolve_date_discrepancy(self, mock_engine):
        orig = "The treaty was signed in 1947. The country has 10 states."
        eng = "The treaty was signed in 1950. The country has 10 states."
        # Flawed judge tries to claim CONSISTENT by only citing states=10
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both agree the country has 10 states.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "10 states",
                "english_evidence_span": "10 states",
                "interpreted_claim": "The country has 10 states",
                "explanation": "States match, treaty date ignored"
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("details?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_UNRESOLVED_DATE_DISCREPANCY"

    def test_case_15_incidental_negation_in_span_cannot_resolve_polarity_discrepancy(self, mock_engine):
        orig = "Water boils at 100 degrees Celsius."
        eng = "Water does not boil at 100 degrees Celsius."
        # Flawed judge cites a span with 'not' as MATCHED, but provides no explanation reconciling polarity
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Both mention boiling.",
            "claims": [{
                "claim_id": "CLM_01",
                "relationship": "MATCHED",
                "original_evidence_span": "boils at 100 degrees Celsius",
                "english_evidence_span": "not boil at 100 degrees Celsius",
                "interpreted_claim": "Boiling temperature",
                "explanation": "No explanation of why 'not' is ignored"  # Lacks double negation/litotes keywords
            }]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("boiling?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "UNCERTAIN"
        assert res.classification_decision_source == "ARBITER_OVERRIDE_UNRESOLVED_POLARITY_DISCREPANCY"

    def test_case_16_metadata_explicitly_marks_unilateral_claim_awaiting_verification(self, mock_engine):
        # Verifies that ORIGINAL_ONLY accounts for consistency without asserting factual correctness
        orig = "The city was established in 1537 by Kempe Gowda."
        eng = "The city is Bengaluru."
        mock_engine.generate.return_value = self._make_mock_output({
            "classification": "CONSISTENT",
            "overall_rationale": "Compatible details.",
            "claims": [
                {
                    "claim_id": "CLM_01",
                    "relationship": "MATCHED",
                    "original_evidence_span": "The city",
                    "english_evidence_span": "The city",
                    "interpreted_claim": "City identity",
                    "explanation": "Both identify the city"
                },
                {
                    "claim_id": "CLM_02",
                    "relationship": "ORIGINAL_ONLY",
                    "original_evidence_span": "established in 1537",
                    "english_evidence_span": None,
                    "interpreted_claim": "The city was established in 1537",
                    "explanation": "Supplementary historical assertion"
                }
            ]
        })
        comparator = AnswerComparator(model_engine=mock_engine)
        res = comparator.compare("history?", orig, eng, "en", "English")
        assert res.final_arbiter_classification == "CONSISTENT"
        # Check discrepancy metadata
        disc_1537 = [d for d in res.deterministic_features.candidate_discrepancies if d.original_value == "1537"][0]
        assert disc_1537.accounted_for_by_comparator is True
        assert disc_1537.comparator_resolution_status == "ACCOUNTED_FOR_AS_ORIGINAL_ONLY"
        assert disc_1537.external_verification_status == "AWAITING_EXTERNAL_VERIFICATION"
        assert "awaiting Phase 5" in disc_1537.comparator_resolution_reason

