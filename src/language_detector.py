"""
Language Detection Engine for Cross-Lingual Calibration.

Supported Languages:
- English (en)
- Kannada (kn)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Malayalam (ml)

Architectural Principles:
1. Clearly distinguish deterministic Unicode script identification from language confidence.
2. For Indic languages, Unicode blocks provide deterministic script isolation; confidence
   reflects the proportional dominance of that script over alphabetic characters.
3. For Latin-script inputs, evaluate character frequency, absence of foreign diacritics,
   word-level lexical distribution, and common English n-gram features.
4. Explicitly represent uncertainty (`is_uncertain=True`, `requires_clarification=True`)
   for short inputs, conflicting scripts, or low-evidence Latin text.
5. Provide strict manual override functionality.
"""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Dict, Optional, Set, Tuple, Any

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}

# Unicode codepoint ranges for Indic scripts
INDIC_SCRIPT_RANGES = {
    "hi": ("Devanagari", 0x0900, 0x097F),
    "ta": ("Tamil", 0x0B80, 0x0BFF),
    "te": ("Telugu", 0x0C00, 0x0C7F),
    "kn": ("Kannada", 0x0C80, 0x0CFF),
    "ml": ("Malayalam", 0x0D00, 0x0D7F),
}

# High-frequency English functional words (closed-class tokens: question words, auxiliary verbs,
# pronouns, prepositions, conjunctions, common determiners)
ENGLISH_LEXICON: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
    "these", "give", "day", "most", "us", "is", "are", "was", "were", "been", "has", "had",
    "does", "did", "where", "why", "whose", "whom", "which", "capital", "city", "state",
    "country", "name", "explain", "describe", "answer", "question", "population", "river",
    "mountain", "king", "history", "science", "math", "true", "false", "difference", "between"
}

# English typical bigrams (top letter transitions in English prose)
COMMON_ENGLISH_BIGRAMS: Set[str] = {
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
    "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le"
}


@dataclass
class DetectionResult:
    raw_text: str
    language_code: Optional[str]
    language_name: Optional[str]
    script_name: str
    confidence: float
    is_uncertain: bool
    requires_clarification: bool
    detection_method: str
    is_overridden: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class LanguageDetector:
    """
    Decoupled language detector distinguishing deterministic script identification
    from empirical language confidence.
    """

    def __init__(self, min_alphabetic_chars: int = 3, script_dominance_threshold: float = 0.50):
        self.min_alphabetic_chars = min_alphabetic_chars
        self.script_dominance_threshold = script_dominance_threshold

    def detect(self, text: str, override_lang: Optional[str] = None) -> DetectionResult:
        """
        Detects language of input text or applies manual override.
        """
        raw_text = text if text is not None else ""
        
        # Check manual override first
        if override_lang is not None:
            return self._apply_override(raw_text, override_lang)

        stripped = raw_text.strip()
        if not stripped:
            return DetectionResult(
                raw_text=raw_text,
                language_code=None,
                language_name=None,
                script_name="Empty",
                confidence=0.0,
                is_uncertain=True,
                requires_clarification=True,
                detection_method="empty_input",
                details={"reason": "Input string is empty or whitespace only"}
            )

        # Count character distributions across scripts
        script_counts: Dict[str, int] = {lang: 0 for lang in INDIC_SCRIPT_RANGES}
        script_counts["latin"] = 0
        script_counts["other_alphabetic"] = 0
        total_alphabetic = 0

        for ch in stripped:
            if not ch.isalpha():
                continue
            
            total_alphabetic += 1
            cp = ord(ch)
            
            # Check Indic blocks
            matched_indic = False
            for lang_code, (_, start, end) in INDIC_SCRIPT_RANGES.items():
                if start <= cp <= end:
                    script_counts[lang_code] += 1
                    matched_indic = True
                    break
            
            if matched_indic:
                continue

            # Check Basic Latin
            if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
                script_counts["latin"] += 1
            else:
                script_counts["other_alphabetic"] += 1

        # Insufficient alphabetic characters (e.g. "12345", "???", "A")
        if total_alphabetic < self.min_alphabetic_chars:
            return DetectionResult(
                raw_text=raw_text,
                language_code=None,
                language_name=None,
                script_name="Indeterminate",
                confidence=0.0,
                is_uncertain=True,
                requires_clarification=True,
                detection_method="insufficient_evidence",
                details={
                    "total_alphabetic": total_alphabetic,
                    "min_required": self.min_alphabetic_chars,
                    "reason": f"Fewer than {self.min_alphabetic_chars} alphabetic characters"
                }
            )

        # Check for unsupported alphabetic scripts (Cyrillic, Arabic, Han, etc.)
        other_ratio = script_counts["other_alphabetic"] / total_alphabetic
        if other_ratio > 0.40:
            return DetectionResult(
                raw_text=raw_text,
                language_code=None,
                language_name=None,
                script_name="Unsupported",
                confidence=0.0,
                is_uncertain=True,
                requires_clarification=True,
                detection_method="unsupported_script",
                details={
                    "other_ratio": round(other_ratio, 3),
                    "reason": "Dominant script is not among the 6 supported languages"
                }
            )

        # Evaluate Indic Script Dominance
        for lang_code, (script_label, _, _) in INDIC_SCRIPT_RANGES.items():
            count = script_counts[lang_code]
            ratio = count / total_alphabetic
            if ratio >= self.script_dominance_threshold:
                # Deterministic script match. Confidence is the script purity ratio.
                return DetectionResult(
                    raw_text=raw_text,
                    language_code=lang_code,
                    language_name=SUPPORTED_LANGUAGES[lang_code],
                    script_name=script_label,
                    confidence=round(ratio, 4),
                    is_uncertain=False,
                    requires_clarification=False,
                    detection_method="deterministic_indic_script",
                    details={
                        "script_purity_ratio": round(ratio, 4),
                        "script_char_count": count,
                        "total_alphabetic": total_alphabetic,
                    }
                )

        # Evaluate Latin Script (English detection)
        latin_count = script_counts["latin"]
        latin_ratio = latin_count / total_alphabetic
        if latin_ratio >= self.script_dominance_threshold:
            return self._detect_english(raw_text, latin_count, total_alphabetic)

        # Conflicting or mixed scripts without clear dominance
        return DetectionResult(
            raw_text=raw_text,
            language_code=None,
            language_name=None,
            script_name="Mixed_Conflicting",
            confidence=0.0,
            is_uncertain=True,
            requires_clarification=True,
            detection_method="mixed_script_conflict",
            details={
                "script_counts": script_counts,
                "total_alphabetic": total_alphabetic,
                "reason": "No single script achieved dominance threshold"
            }
        )

    def _detect_english(self, text: str, latin_count: int, total_alphabetic: int) -> DetectionResult:
        """
        Analyzes Latin text to distinguish English from other Latin languages or random characters.
        Uses lexical overlap, diacritic checks, and bigram frequency.
        """
        # Tokenize Latin words
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return DetectionResult(
                raw_text=text,
                language_code=None,
                language_name=None,
                script_name="Latin",
                confidence=0.0,
                is_uncertain=True,
                requires_clarification=True,
                detection_method="latin_no_words",
                details={"reason": "No valid Latin words found"}
            )

        # Check for Latin diacritics / accents that do not exist in standard English
        normalized = unicodedata.normalize('NFKD', text)
        combining_accents = [ch for ch in normalized if unicodedata.combining(ch)]
        if len(combining_accents) > 1:
            # Significant accents indicate Spanish, French, German, etc.
            return DetectionResult(
                raw_text=text,
                language_code=None,
                language_name=None,
                script_name="Latin_Extended",
                confidence=0.20,
                is_uncertain=True,
                requires_clarification=True,
                detection_method="latin_foreign_diacritics",
                details={
                    "accent_count": len(combining_accents),
                    "reason": "Contains non-English Latin diacritics / accents"
                }
            )

        # Lexical score: proportion of words matching common English vocabulary
        matching_words = [w for w in words if w in ENGLISH_LEXICON]
        lexical_ratio = len(matching_words) / len(words)

        # Bigram transition score
        clean_text = "".join(words)
        bigrams = [clean_text[i:i+2] for i in range(len(clean_text) - 1)]
        if bigrams:
            matching_bigrams = [bg for bg in bigrams if bg in COMMON_ENGLISH_BIGRAMS]
            bigram_ratio = len(matching_bigrams) / len(bigrams)
        else:
            bigram_ratio = 0.5

        # Script purity
        script_purity = latin_count / total_alphabetic

        # Composite statistical confidence score
        # Weighted: 50% lexical overlap, 30% bigram transition structure, 20% script purity
        if len(words) >= 3:
            # Multi-word input
            confidence = (0.50 * lexical_ratio) + (0.30 * bigram_ratio) + (0.20 * script_purity)
            # If at least one strong English function word is present, boost threshold
            if lexical_ratio >= 0.20 or (lexical_ratio > 0 and bigram_ratio >= 0.35):
                is_uncertain = False
            else:
                is_uncertain = confidence < 0.45
        else:
            # Short 1-2 word input (e.g., "Bangalore", "Karnataka capital")
            if any(w in ENGLISH_LEXICON for w in words):
                confidence = 0.85
                is_uncertain = False
            elif bigram_ratio >= 0.40:
                confidence = 0.65
                is_uncertain = False
            else:
                confidence = 0.35
                is_uncertain = True

        requires_clarification = is_uncertain

        return DetectionResult(
            raw_text=text,
            language_code="en" if not is_uncertain else None,
            language_name="English" if not is_uncertain else None,
            script_name="Latin",
            confidence=round(confidence, 4),
            is_uncertain=is_uncertain,
            requires_clarification=requires_clarification,
            detection_method="statistical_latin_profile",
            details={
                "word_count": len(words),
                "matching_lexicon_count": len(matching_words),
                "lexical_ratio": round(lexical_ratio, 4),
                "bigram_ratio": round(bigram_ratio, 4),
                "script_purity": round(script_purity, 4),
            }
        )

    def _apply_override(self, raw_text: str, override_lang: str) -> DetectionResult:
        """
        Applies a validated manual language override.
        """
        code = override_lang.lower().strip()
        if code not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported manual override language '{code}'. "
                f"Must be one of: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        return DetectionResult(
            raw_text=raw_text,
            language_code=code,
            language_name=SUPPORTED_LANGUAGES[code],
            script_name="ManualOverride",
            confidence=1.0,
            is_uncertain=False,
            requires_clarification=False,
            detection_method="manual_override",
            is_overridden=True,
            details={"manual_code": code, "manual_name": SUPPORTED_LANGUAGES[code]}
        )
