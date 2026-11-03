"""
Unit tests for src/language_detector.py.
Executes completely offline without loading Gemma or requiring CUDA.
"""

import pytest
from src.language_detector import LanguageDetector, SUPPORTED_LANGUAGES


@pytest.fixture
def detector():
    return LanguageDetector()


class TestLanguageDetectionSixLanguages:
    """Verify accurate detection across all six target languages."""

    def test_english_detection(self, detector):
        text = "What is the capital city of Karnataka?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "en"
        assert res.language_name == "English"
        assert res.script_name == "Latin"
        assert res.confidence > 0.60
        assert not res.is_uncertain
        assert not res.requires_clarification
        assert res.detection_method == "statistical_latin_profile"

    def test_kannada_detection(self, detector):
        text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "kn"
        assert res.language_name == "Kannada"
        assert res.script_name == "Kannada"
        assert res.confidence >= 0.95
        assert not res.is_uncertain
        assert not res.requires_clarification
        assert res.detection_method == "deterministic_indic_script"

    def test_hindi_detection(self, detector):
        text = "कर्नाटक की राजधानी क्या है?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "hi"
        assert res.language_name == "Hindi"
        assert res.script_name == "Devanagari"
        assert res.confidence >= 0.95
        assert not res.is_uncertain
        assert not res.requires_clarification

    def test_tamil_detection(self, detector):
        text = "கர்நாடகாவின் தலைநகரம் எது?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "ta"
        assert res.language_name == "Tamil"
        assert res.script_name == "Tamil"
        assert res.confidence >= 0.95
        assert not res.is_uncertain

    def test_telugu_detection(self, detector):
        text = "కర్ణాటక రాజధాని ఏది?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "te"
        assert res.language_name == "Telugu"
        assert res.script_name == "Telugu"
        assert res.confidence >= 0.95
        assert not res.is_uncertain

    def test_malayalam_detection(self, detector):
        text = "കർണാടകയുടെ തലസ്ഥാനം ഏതാണ്?"
        res = detector.detect(text)
        assert res.raw_text == text
        assert res.language_code == "ml"
        assert res.language_name == "Malayalam"
        assert res.script_name == "Malayalam"
        assert res.confidence >= 0.95
        assert not res.is_uncertain


class TestAmbiguityAndEdgeCases:
    """Verify proper handling of short, ambiguous, or unsupported text."""

    def test_empty_string(self, detector):
        res = detector.detect("   ")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True
        assert res.detection_method == "empty_input"

    def test_insufficient_characters(self, detector):
        res = detector.detect("a?")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True
        assert res.detection_method == "insufficient_evidence"

    def test_numbers_only(self, detector):
        res = detector.detect("1234567 890")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True

    def test_unsupported_cyrillic(self, detector):
        res = detector.detect("Какая столица Карнатаки?")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True
        assert res.detection_method == "unsupported_script"

    def test_unsupported_chinese(self, detector):
        res = detector.detect("卡纳塔克邦的首都是什么？")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True

    def test_latin_with_accents_flagged_uncertain(self, detector):
        # French with accents: "Comment ça va à Paris?"
        res = detector.detect("Comment ça va à Paris?")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True
        assert res.detection_method == "latin_foreign_diacritics"

    def test_punctuation_only(self, detector):
        res = detector.detect("?!? --- ... !!!")
        assert res.language_code is None
        assert res.is_uncertain is True
        assert res.requires_clarification is True

    def test_short_english_query(self, detector):
        res = detector.detect("Who is Einstein?")
        assert res.language_code == "en"
        assert not res.is_uncertain

    def test_code_mixed_indic_dominance(self, detector):
        # Kannada sentence with English entity "Bengaluru"
        text = "Bengaluru ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ"
        res = detector.detect(text)
        assert res.language_code == "kn"
        assert not res.is_uncertain


class TestManualOverride:
    """Verify manual language override mechanism."""

    def test_valid_override(self, detector):
        text = "123"  # Ambiguous text
        res = detector.detect(text, override_lang="kn")
        assert res.language_code == "kn"
        assert res.language_name == "Kannada"
        assert res.is_overridden is True
        assert res.confidence == 1.0
        assert not res.is_uncertain
        assert not res.requires_clarification

    def test_invalid_override_raises_value_error(self, detector):
        with pytest.raises(ValueError) as excinfo:
            detector.detect("Sample text", override_lang="french")
        assert "Unsupported manual override" in str(excinfo.value)
