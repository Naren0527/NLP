"""
Answer Comparison Engine for Cross-Lingual Calibration.

Compares an original-language answer with its independently generated English counterpart
for the same semantic question to evaluate factual equivalence.

Core Principles:
1. Symmetrical Comparison: Neither answer is ground truth; English has zero priority.
2. Evidence Grounding: Every claim must cite exact substrings from the source answers.
   Spans are programmatically verified via exact substring containment.
3. Decoupled Architecture:
   - DeterministicFeatureExtractor: Normalizes digits and extracts candidate differences with context.
   - ModelComparisonJudge: Uses Gemma 3 4B Instruct via isolated comparison prompt.
   - ClassificationArbiter: Enforces evidence grounding and the formal Decision Table.
"""

from dataclasses import dataclass, asdict, field
import json
import os
import re
import time
import unicodedata
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any

import yaml

from src.language_detector import SUPPORTED_LANGUAGES
from src.confidence import normalize_unicode_digits


# Language-specific negation tokens
NEGATION_TOKENS: Dict[str, Set[str]] = {
    "en": {"not", "never", "no", "neither", "nor", "none", "without", "n't"},
    "kn": {"ಅಲ್ಲ", "ಇಲ್ಲ", "ಬೇಡ", "ಆಗದು", "ಕೂಡದು", "ವಲ್ಲ"},
    "hi": {"नहीं", "ना", "मत", "बगैर", "बिना"},
    "ta": {"இல்லை", "அல்ல", "கூடாது", "வேண்டாம்"},
    "te": {"కాదు", "లేదు", "వద్దు", "రాదు"},
    "ml": {"അല്ല", "ഇല്ല", "പാടില്ല", "വേണ്ട"},
}

ALLOWED_RELATIONSHIPS = {
    "MATCHED", "CONTRADICTORY", "ORIGINAL_ONLY", "ENGLISH_ONLY", "AMBIGUOUS"
}


@dataclass
class EvidenceGroundedClaim:
    claim_id: str
    relationship: str
    original_evidence_span: Optional[str]
    english_evidence_span: Optional[str]
    interpreted_claim: str
    explanation: str
    is_original_evidence_valid: bool = False
    is_english_evidence_valid: bool = False
    validation_error: Optional[str] = None
    external_verification_status: str = "AWAITING_EXTERNAL_VERIFICATION"


@dataclass
class CandidateDiscrepancy:
    category: str                      # 'NUMERICAL', 'DATE', 'POLARITY', 'ENTITY'
    original_value: Optional[str]
    original_context: Optional[str]
    english_value: Optional[str]
    english_context: Optional[str]
    potential_attribute: Optional[str]

    # Comparator Accounting Metadata (Cross-Answer Consistency Evaluation)
    accounted_for_by_comparator: bool = False
    resolving_claim_id: Optional[str] = None
    comparator_resolution_status: str = "UNRESOLVED"
    comparator_resolution_reason: Optional[str] = None
    is_unresolved: bool = True
    judge_assessment: Optional[str] = None

    # External Factual Verification Metadata (Phase 5 Concept - Separate from Consistency)
    external_verification_status: str = "AWAITING_EXTERNAL_VERIFICATION"

    @property
    def resolved_by_judge(self) -> bool:
        return self.accounted_for_by_comparator

    @property
    def resolution_status(self) -> str:
        return self.comparator_resolution_status

    @property
    def resolution_reason(self) -> Optional[str]:
        return self.comparator_resolution_reason


@dataclass
class DeterministicFeatureSummary:
    extractor_version: str
    candidate_discrepancies: List[CandidateDiscrepancy]
    has_numerical_candidates: bool
    has_date_candidates: bool
    has_polarity_candidates: bool
    has_entity_candidates: bool


@dataclass
class ComparisonResult:
    # 1. Identifiers and Text References
    comparison_id: str
    run_id: str
    original_record_id: Optional[str]
    english_record_id: Optional[str]
    original_language_code: str
    original_language_name: str
    english_language_code: str
    original_answer_text: str
    english_answer_text: str

    # 2. Dual Classification & Provenance
    raw_judge_classification: str      # 'CONSISTENT', 'INCONSISTENT', 'UNCERTAIN', or 'MALFORMED'
    final_arbiter_classification: str  # Exactly one of: 'CONSISTENT', 'INCONSISTENT', 'UNCERTAIN'
    classification_decision_source: str# 'JUDGE_ACCEPTED', 'ARBITER_OVERRIDE_UNGROUNDED_EVIDENCE', etc.
    arbiter_override_reason: Optional[str]
    overall_rationale: str

    # 3. Evidence-Grounded Claims (Validated Substrings)
    claims: List[EvidenceGroundedClaim]
    evidence_validation_status: str    # 'ALL_EVIDENCE_GROUNDED', 'PARTIAL_EVIDENCE_UNGROUNDED', 'FABRICATED_EVIDENCE_DETECTED', etc.

    # 4. Grouped Claim Views
    matched_claims: List[EvidenceGroundedClaim]
    contradictory_claims: List[EvidenceGroundedClaim]
    original_only_claims: List[EvidenceGroundedClaim]
    english_only_claims: List[EvidenceGroundedClaim]
    ambiguous_claims: List[EvidenceGroundedClaim]

    # 5. Deterministic Feature Extraction Record
    deterministic_features: DeterministicFeatureSummary

    # 6. Judge Execution Provenance
    comparison_method: str
    prompt_version: str
    judge_model_id: str
    judge_model_revision: str
    judge_generation_config: Dict[str, Any]
    raw_judge_response: str
    execution_status: str
    execution_time_seconds: float


class DeterministicFeatureExtractor:
    """
    Extracts candidate differences (numbers, dates, negation) without declaring
    semantic contradictions, preserving contextual clauses.
    """
    VERSION = "v1.0_deterministic_extractor"

    @classmethod
    def extract_numbers_with_context(cls, text: str, window: int = 35) -> List[Tuple[str, str]]:
        """Extracts normalized numbers along with surrounding context."""
        normalized = normalize_unicode_digits(text)
        matches = []
        for match in re.finditer(r"\b\d+(?:[\.,]\d+)?\b", normalized):
            val = match.group(0)
            start = max(0, match.start() - window)
            end = min(len(normalized), match.end() + window)
            context = normalized[start:end].strip()
            matches.append((val, context))
        return matches

    @classmethod
    def extract_potential_attribute(cls, val: str, ctx: str) -> Optional[str]:
        """Extracts the immediate noun/attribute associated with a number in its context."""
        stopwords = {
            "in", "on", "at", "by", "of", "to", "for", "from", "with", "as", "is", "was", "are", "were",
            "the", "a", "an", "and", "or", "it", "its", "has", "have", "had", "been", "be"
        }
        # Look for word immediately following number
        m_after = re.search(r"\b" + re.escape(val) + r"\s+([a-zA-Z\u0900-\u0D7F]+)", ctx)
        if m_after:
            w = m_after.group(1).lower()
            if len(w) >= 3 and w not in stopwords:
                return w
        # Look for word immediately preceding number
        m_before = re.search(r"([a-zA-Z\u0900-\u0D7F]+)\s+" + re.escape(val) + r"\b", ctx)
        if m_before:
            w = m_before.group(1).lower()
            if len(w) >= 3 and w not in stopwords:
                return w
        return None

    @classmethod
    def extract_negations_with_context(cls, text: str, lang_code: str, window: int = 35) -> List[Tuple[str, str]]:
        """Extracts negation tokens and their grammatical context."""
        neg_set = NEGATION_TOKENS.get(lang_code, NEGATION_TOKENS["en"])
        matches = []
        # Split by whitespace and punctuation to safely handle Indic scripts (avoiding \b splitting on virama/Mn characters)
        words_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[^\s.,;:!?()\[\]{}'\"।॥]+", text)]
        for word, start_pos, end_pos in words_with_spans:
            w_lower = word.lower()
            matched = False
            if w_lower in neg_set:
                matched = True
            elif lang_code in ("kn", "ta", "te", "ml"):
                # Dravidian agglutinative suffixes (e.g. ಸತ್ಯವಲ್ಲ -> ends with ವಲ್ಲ or ಅಲ್ಲ)
                for neg in neg_set:
                    if len(neg) >= 2 and w_lower.endswith(neg):
                        matched = True
                        break
            elif lang_code == "en" and (w_lower.endswith("n't") or w_lower in neg_set):
                matched = True
            if matched:
                ctx_start = max(0, start_pos - window)
                ctx_end = min(len(text), end_pos + window)
                context = text[ctx_start:ctx_end].strip()
                matches.append((word, context))
        return matches

    @classmethod
    def analyze(
        cls,
        original_text: str,
        english_text: str,
        orig_lang: str
    ) -> DeterministicFeatureSummary:
        candidates: List[CandidateDiscrepancy] = []

        # 1. Number comparison candidates
        orig_nums = cls.extract_numbers_with_context(original_text)
        eng_nums = cls.extract_numbers_with_context(english_text)

        orig_num_vals = {n[0] for n in orig_nums}
        eng_num_vals = {n[0] for n in eng_nums}

        if orig_num_vals != eng_num_vals:
            # Different numbers exist: record as candidate differences (NOT automatic contradiction)
            for oval, octx in orig_nums:
                if oval not in eng_num_vals:
                    candidates.append(CandidateDiscrepancy(
                        category="NUMERICAL",
                        original_value=oval,
                        original_context=octx,
                        english_value=None,
                        english_context=None,
                        potential_attribute=cls.extract_potential_attribute(oval, octx)
                    ))
            for eval_val, ectx in eng_nums:
                if eval_val not in orig_num_vals:
                    candidates.append(CandidateDiscrepancy(
                        category="NUMERICAL",
                        original_value=None,
                        original_context=None,
                        english_value=eval_val,
                        english_context=ectx,
                        potential_attribute=cls.extract_potential_attribute(eval_val, ectx)
                    ))

        # 2. Date/Year comparison candidates (4-digit years)
        orig_years = {n[0] for n in orig_nums if re.match(r"^(18|19|20)\d{2}$", n[0])}
        eng_years = {n[0] for n in eng_nums if re.match(r"^(18|19|20)\d{2}$", n[0])}
        if orig_years and eng_years and orig_years != eng_years:
            candidates.append(CandidateDiscrepancy(
                category="DATE",
                original_value=",".join(sorted(orig_years)),
                original_context=original_text[:80],
                english_value=",".join(sorted(eng_years)),
                english_context=english_text[:80],
                potential_attribute="year"
            ))

        # 3. Negation presence candidates
        orig_negs = cls.extract_negations_with_context(original_text, orig_lang)
        eng_negs = cls.extract_negations_with_context(english_text, "en")

        has_orig_neg = len(orig_negs) > 0
        has_eng_neg = len(eng_negs) > 0

        if has_orig_neg != has_eng_neg:
            candidates.append(CandidateDiscrepancy(
                category="POLARITY",
                original_value=orig_negs[0][0] if orig_negs else None,
                original_context=orig_negs[0][1] if orig_negs else None,
                english_value=eng_negs[0][0] if eng_negs else None,
                english_context=eng_negs[0][1] if eng_negs else None,
                potential_attribute="polarity_candidate"
            ))

        return DeterministicFeatureSummary(
            extractor_version=cls.VERSION,
            candidate_discrepancies=candidates,
            has_numerical_candidates=any(c.category == "NUMERICAL" for c in candidates),
            has_date_candidates=any(c.category == "DATE" for c in candidates),
            has_polarity_candidates=any(c.category == "POLARITY" for c in candidates),
            has_entity_candidates=any(c.category == "ENTITY" for c in candidates),
        )


class EvidenceValidator:
    """
    Validates claim evidence spans against the approved relationship contract:
    - MATCHED: original required, english required.
    - CONTRADICTORY: original required, english required.
    - ORIGINAL_ONLY: original required, english must be null.
    - ENGLISH_ONLY: original must be null, english required.
    - AMBIGUOUS: at least one required.
    Validates exact substring containment against source answers.
    """

    @classmethod
    def validate_claim(
        cls,
        claim: EvidenceGroundedClaim,
        original_text: str,
        english_text: str
    ) -> Tuple[bool, bool, Optional[str]]:
        rel = claim.relationship.upper()
        if rel not in ALLOWED_RELATIONSHIPS:
            return False, False, f"INVALID_RELATIONSHIP_CATEGORY: '{rel}'"

        if not claim.interpreted_claim or not claim.interpreted_claim.strip():
            return False, False, "EMPTY_INTERPRETED_CLAIM"

        orig_span = claim.original_evidence_span
        eng_span = claim.english_evidence_span

        # Contract Checks
        if rel in ("MATCHED", "CONTRADICTORY"):
            if not orig_span or not orig_span.strip():
                return False, False, f"MISSING_ORIGINAL_EVIDENCE_FOR_{rel}"
            if not eng_span or not eng_span.strip():
                return False, False, f"MISSING_ENGLISH_EVIDENCE_FOR_{rel}"

        elif rel == "ORIGINAL_ONLY":
            if not orig_span or not orig_span.strip():
                return False, False, "MISSING_ORIGINAL_EVIDENCE_FOR_ORIGINAL_ONLY"
            if eng_span is not None and eng_span.strip():
                return False, False, "ENGLISH_EVIDENCE_MUST_BE_NULL_FOR_ORIGINAL_ONLY"

        elif rel == "ENGLISH_ONLY":
            if orig_span is not None and orig_span.strip():
                return False, False, "ORIGINAL_EVIDENCE_MUST_BE_NULL_FOR_ENGLISH_ONLY"
            if not eng_span or not eng_span.strip():
                return False, False, "MISSING_ENGLISH_EVIDENCE_FOR_ENGLISH_ONLY"

        elif rel == "AMBIGUOUS":
            has_orig = bool(orig_span and orig_span.strip())
            has_eng = bool(eng_span and eng_span.strip())
            if not has_orig and not has_eng:
                return False, False, "MISSING_EVIDENCE_FOR_AMBIGUOUS_CLAIM"

        # Exact Substring Containment Checks
        orig_valid = True
        if orig_span is not None:
            clean_orig = orig_span.strip()
            if clean_orig not in original_text:
                return False, True if eng_span is None else (eng_span.strip() in english_text), (
                    f"ORIGINAL_SPAN_NOT_FOUND_IN_SOURCE: '{orig_span}'"
                )

        eng_valid = True
        if eng_span is not None:
            clean_eng = eng_span.strip()
            if clean_eng not in english_text:
                return orig_valid, False, (
                    f"ENGLISH_SPAN_NOT_FOUND_IN_SOURCE: '{eng_span}'"
                )

        return True, True, None


class ClassificationArbiter:
    """
    Arbitrates the final classification using the formal Decision Table.
    Enforces evidence grounding and resolves conflicts between deterministic signals and the judge.
    """

    @classmethod
    def does_claim_address_discrepancy(
        cls,
        claim: EvidenceGroundedClaim,
        disc: CandidateDiscrepancy
    ) -> bool:
        """
        Determines whether a claim's evidence spans or text explicitly refer to
        the candidate discrepancy's values and its specific contextual attribute.
        Guarantees:
        1. An unrelated claim (e.g. city name) will return False for a date/number discrepancy.
        2. A numerical candidate cannot be resolved by a claim mentioning the same number for a different attribute.
        3. A date candidate cannot be resolved by an unrelated historical claim.
        4. A polarity candidate cannot be resolved merely because a negation token appears incidentally in a cited span.
        """
        orig_span = (claim.original_evidence_span or "").lower()
        eng_span = (claim.english_evidence_span or "").lower()
        claim_text = (claim.interpreted_claim or "").lower() + " " + (claim.explanation or "").lower()

        if disc.category in ("NUMERICAL", "DATE"):
            oval = (disc.original_value or "").lower()
            eval_val = (disc.english_value or "").lower()

            orig_has_val = bool(oval and (oval in orig_span or oval in claim_text))
            eng_has_val = bool(eval_val and (eval_val in eng_span or eval_val in claim_text))

            attr = (disc.potential_attribute or "").lower()
            orig_ctx = (disc.original_context or "").lower()
            eng_ctx = (disc.english_context or "").lower()

            attr_in_claim = False
            if attr and len(attr) >= 3 and (attr in claim_text or attr in orig_span or attr in eng_span):
                attr_in_claim = True

            # If both values exist (e.g. 60 vs 100, or 1947 vs 1950)
            if oval and eval_val:
                # Must either explicitly mention both values, or mention at least one value with the matching attribute
                if orig_has_val and eng_has_val:
                    return True
                if (orig_has_val or eng_has_val) and attr_in_claim:
                    return True
                return False

            elif oval:
                # Unilateral number in original
                if orig_has_val:
                    # If candidate has a specific attribute (e.g. "campuses"), claim must NOT describe a conflicting attribute
                    if attr and not attr_in_claim:
                        return False
                    return True
                return False

            elif eval_val:
                # Unilateral number in English
                if eng_has_val:
                    if attr and not attr_in_claim:
                        return False
                    return True
                return False

            return False

        elif disc.category == "POLARITY":
            neg_val = (disc.original_value or disc.english_value or "").lower()
            # Negation token must actually be cited in one of the evidence spans
            if neg_val and (neg_val in orig_span or neg_val in eng_span):
                # If claim is MATCHED, it must explicitly explain the reconciliation of the polarity marker
                if claim.relationship == "MATCHED":
                    for kw in ("double", "litotes", "reconcil", "idiom", "negat", "affirm", "equival", "rhetorical"):
                        if kw in claim_text:
                            return True
                    return False
                return True
            # Or explanation explicitly cites the specific negation token as a whole word
            if neg_val and re.search(r"\b" + re.escape(neg_val) + r"\b", claim_text):
                if claim.relationship == "MATCHED":
                    for kw in ("double", "litotes", "reconcil", "idiom", "negat", "affirm", "equival", "rhetorical"):
                        if kw in claim_text:
                            return True
                    return False
                return True
            return False

        return False

    @classmethod
    def resolve_candidate_discrepancies(
        cls,
        deterministic: DeterministicFeatureSummary,
        validated_claims: List[EvidenceGroundedClaim]
    ) -> List[CandidateDiscrepancy]:
        """
        Matches candidate discrepancies to validated claims using contextual attributes.
        Distinguishes between:
        1. Accounting by comparator (cross-answer consistency assessment)
        2. External factual verification (Phase 5 concept - separate from consistency)
        Updates CandidateDiscrepancy fields in-place.
        Returns list of unresolved discrepancies.
        """
        unresolved = []
        for disc in deterministic.candidate_discrepancies:
            addressing_claims = [c for c in validated_claims if cls.does_claim_address_discrepancy(c, disc)]

            if not addressing_claims:
                disc.accounted_for_by_comparator = False
                disc.resolving_claim_id = None
                disc.comparator_resolution_status = "UNRESOLVED"
                disc.comparator_resolution_reason = (
                    f"Candidate discrepancy {disc.category} (orig='{disc.original_value}', eng='{disc.english_value}') "
                    f"was omitted or ignored by the judge."
                )
                disc.is_unresolved = True
                disc.judge_assessment = "OMITTED_BY_JUDGE"
                disc.external_verification_status = "AWAITING_EXTERNAL_VERIFICATION"
                unresolved.append(disc)
                continue

            primary_claim = addressing_claims[0]
            disc.resolving_claim_id = primary_claim.claim_id
            disc.accounted_for_by_comparator = True
            disc.judge_assessment = f"ADDRESSED_IN_{primary_claim.relationship}"
            disc.external_verification_status = "AWAITING_EXTERNAL_VERIFICATION"

            if primary_claim.relationship == "CONTRADICTORY":
                disc.comparator_resolution_status = "ACCOUNTED_FOR_AS_CONTRADICTION"
                disc.comparator_resolution_reason = (
                    f"Candidate discrepancy identified as a material contradiction in [{primary_claim.claim_id}]."
                )
                disc.is_unresolved = False

            elif primary_claim.relationship in ("ORIGINAL_ONLY", "ENGLISH_ONLY"):
                disc.comparator_resolution_status = f"ACCOUNTED_FOR_AS_{primary_claim.relationship}"
                disc.comparator_resolution_reason = (
                    f"Unilateral detail accounted for by comparator as {primary_claim.relationship} in [{primary_claim.claim_id}] "
                    f"(proves cross-answer accounting, NOT factual correctness; factual truth remains unverified awaiting Phase 5)."
                )
                disc.is_unresolved = False

            elif primary_claim.relationship == "MATCHED":
                disc.comparator_resolution_status = "ACCOUNTED_FOR_AS_MATCHED"
                disc.comparator_resolution_reason = (
                    f"Candidate discrepancy reconciled within MATCHED claim [{primary_claim.claim_id}]."
                )
                disc.is_unresolved = False

            elif primary_claim.relationship == "AMBIGUOUS":
                disc.comparator_resolution_status = "REJECTED_RESOLUTION_AMBIGUOUS"
                disc.comparator_resolution_reason = (
                    f"Candidate discrepancy is marked AMBIGUOUS in [{primary_claim.claim_id}]."
                )
                disc.is_unresolved = True
                unresolved.append(disc)

            else:
                disc.comparator_resolution_status = "UNRESOLVED"
                disc.comparator_resolution_reason = (
                    f"Candidate discrepancy could not be defensibly resolved by claim [{primary_claim.claim_id}]."
                )
                disc.is_unresolved = True
                unresolved.append(disc)

        return unresolved

    @classmethod
    def arbitrate(
        cls,
        raw_judge_class: str,
        overall_rationale: str,
        validated_claims: List[EvidenceGroundedClaim],
        validation_status: str,
        deterministic: DeterministicFeatureSummary,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Returns: (final_arbiter_classification, decision_source, arbiter_override_reason)
        """
        # Rule 1: Malformed JSON output fallback
        if raw_judge_class == "MALFORMED":
            return (
                "UNCERTAIN",
                "MALFORMED_OUTPUT_FALLBACK",
                "Judge output could not be parsed into valid schema."
            )

        # Rule 2: Evidence validation failure forces UNCERTAIN
        if validation_status != "ALL_EVIDENCE_GROUNDED":
            return (
                "UNCERTAIN",
                "ARBITER_OVERRIDE_UNGROUNDED_EVIDENCE",
                f"Evidence grounding failed ({validation_status}). Spans are not exact source substrings or contract was violated."
            )

        judge_class = raw_judge_class.upper().strip()

        # Step 3: Evaluate and resolve candidate discrepancies against claims
        unresolved_disc = cls.resolve_candidate_discrepancies(deterministic, validated_claims)

        contradictory = [c for c in validated_claims if c.relationship == "CONTRADICTORY"]
        ambiguous = [c for c in validated_claims if c.relationship == "AMBIGUOUS"]

        # Rule B: Ambiguous claims force UNCERTAIN
        # If any material claim is AMBIGUOUS, final classification must be UNCERTAIN
        if ambiguous:
            if judge_class != "UNCERTAIN":
                return (
                    "UNCERTAIN",
                    "ARBITER_OVERRIDE_AMBIGUOUS_CLAIM_PRESENT",
                    f"Material claim(s) {[c.claim_id for c in ambiguous]} marked AMBIGUOUS; cannot declare {judge_class}."
                )
            return (
                "UNCERTAIN",
                "JUDGE_ACCEPTED",
                None
            )

        # Rule A: Contradictory claims force INCONSISTENT
        if contradictory:
            if judge_class == "CONSISTENT":
                return (
                    "INCONSISTENT",
                    "ARBITER_OVERRIDE_CONTRADICTORY_CLAIM_PRESENT",
                    "Judge classified as CONSISTENT, but verified CONTRADICTORY claims exist."
                )
            return (
                "INCONSISTENT",
                "JUDGE_ACCEPTED" if judge_class == "INCONSISTENT" else "ARBITER_OVERRIDE_CONTRADICTORY_CLAIM_PRESENT",
                None if judge_class == "INCONSISTENT" else "Verified CONTRADICTORY claims force INCONSISTENT classification."
            )

        # If judge declared INCONSISTENT without any valid CONTRADICTORY claims
        if judge_class == "INCONSISTENT" and not contradictory:
            return (
                "UNCERTAIN",
                "ARBITER_OVERRIDE_NO_VALID_CONTRADICTION",
                "Judge declared INCONSISTENT but no valid CONTRADICTORY claims with grounded evidence were provided."
            )

        # Rule C & D: Evaluation of CONSISTENT verdict against unresolved discrepancies & omissions
        if judge_class == "CONSISTENT":
            if unresolved_disc:
                cat_set = sorted(list({d.category for d in unresolved_disc}))
                cat_name = cat_set[0] if len(cat_set) == 1 else "DETERMINISTIC"
                if "DATE" in cat_set:
                    cat_name = "DATE"
                elif "POLARITY" in cat_set:
                    cat_name = "POLARITY"
                elif "NUMERICAL" in cat_set:
                    cat_name = "NUMERICAL"
                desc_str = "; ".join([f"{d.category}(orig='{d.original_value}', eng='{d.english_value}')" for d in unresolved_disc])
                return (
                    "UNCERTAIN",
                    f"ARBITER_OVERRIDE_UNRESOLVED_{cat_name}_DISCREPANCY",
                    f"Judge declared CONSISTENT, but candidate discrepancy was left unresolved by judge: {desc_str}."
                )

            # Permitted additional details / matched claims
            return (
                "CONSISTENT",
                "JUDGE_ACCEPTED",
                None
            )

        # Rule 5: Evaluation of UNCERTAIN verdict
        if judge_class == "UNCERTAIN":
            return (
                "UNCERTAIN",
                "JUDGE_ACCEPTED",
                None
            )

        # Fallback for unrecognized class
        return (
            "UNCERTAIN",
            "INVALID_CLASSIFICATION_FALLBACK",
            f"Unrecognized classification '{judge_class}' from judge."
        )


class AnswerComparator:
    """
    Orchestrates deterministic extraction, model judge invocation, evidence validation,
    and final arbitration.
    """

    def __init__(
        self,
        model_engine: Optional[Any] = None,
        prompts_path: str = "configs/prompts.yaml",
        model_config_path: str = "configs/model.yaml",
    ):
        self.model_engine = model_engine
        self.prompts_path = prompts_path
        self.model_config_path = model_config_path
        self.prompts_config = self._load_prompts(prompts_path)
        self.model_config = self._load_prompts(model_config_path)

        comp_gen = self.model_config.get("comparison_generation", {})
        self.judge_generation_config = {
            "max_new_tokens": comp_gen.get("max_new_tokens", 1024),
            "do_sample": comp_gen.get("do_sample", False),
        }

    @staticmethod
    def _load_prompts(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def format_judge_prompt(
        self,
        question: str,
        original_answer: str,
        english_answer: str,
        language_name: str
    ) -> Tuple[str, str]:
        cfg = self.prompts_config.get("prompt_versions", {}).get("comparison_judge", {})
        version = cfg.get("version", "v1.0_comparison_judge")
        template = cfg.get("template")

        if not template:
            template = (
                "Compare the two answers below for factual consistency.\n"
                "Question: {QUESTION}\n"
                "Answer A ({LANGUAGE}): {ORIGINAL_ANSWER}\n"
                "Answer B (English): {ENGLISH_ANSWER}\n"
                "Emit valid JSON with classification and claims."
            )

        prompt = template.format(
            QUESTION=question.strip(),
            ORIGINAL_ANSWER=original_answer.strip(),
            ENGLISH_ANSWER=english_answer.strip(),
            LANGUAGE=language_name.strip()
        )
        return prompt, version

    def parse_judge_json(self, raw_text: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """Extracts and parses JSON object from model output."""
        if not raw_text or not raw_text.strip():
            return {}, False, "EMPTY_RESPONSE"

        # Search for first { to matching }
        cleaned = raw_text.strip()
        # Remove markdown ```json ... ``` wrapper if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Regex search for complete JSON object
        json_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        elif "{" in cleaned:
            # Partial or truncated JSON starting with {
            json_str = cleaned[cleaned.find("{"):].strip()
        else:
            return {}, False, "NO_JSON_OBJECT_FOUND"
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return {}, False, "JSON_NOT_A_DICTIONARY"
            return data, True, None
        except json.JSONDecodeError as err:
            return {}, False, f"JSON_DECODE_ERROR: {str(err)}"

    def compare(
        self,
        question: str,
        original_answer_text: str,
        english_answer_text: str,
        original_language_code: str,
        original_language_name: Optional[str] = None,
        run_id: Optional[str] = None,
        original_record_id: Optional[str] = None,
        english_record_id: Optional[str] = None,
    ) -> ComparisonResult:
        comp_id = f"comp_{uuid.uuid4().hex[:12]}"
        actual_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        lang_code = original_language_code.lower().strip()
        lang_name = original_language_name or SUPPORTED_LANGUAGES.get(lang_code, "Original Language")

        start_time = time.time()

        # Step 1: Deterministic Feature Extraction
        deterministic_summary = DeterministicFeatureExtractor.analyze(
            original_text=original_answer_text,
            english_text=english_answer_text,
            orig_lang=lang_code
        )

        # Step 2: Format and Execute Judge Prompt
        judge_prompt, prompt_version = self.format_judge_prompt(
            question=question,
            original_answer=original_answer_text,
            english_answer=english_answer_text,
            language_name=lang_name
        )

        judge_model_id = "google/gemma-3-4b-it"
        judge_revision = getattr(self.model_engine, "model_revision", "093f9f388b31de276ce2de164bdc2081324b9767")
        judge_config = self.judge_generation_config

        raw_judge_text = ""
        execution_status = "SUCCESS"

        if self.model_engine is not None:
            try:
                gen_out = self.model_engine.generate(
                    prompt=judge_prompt,
                    max_new_tokens=judge_config["max_new_tokens"],
                    apply_template=True
                )
                raw_judge_text = gen_out.answer_text
            except Exception as e:
                execution_status = f"JUDGE_GENERATION_FAILED: {str(e)}"
                raw_judge_text = ""
        else:
            execution_status = "MODEL_ENGINE_NOT_INITIALIZED"

        elapsed_time = round(time.time() - start_time, 4)

        # Step 3: Parse Judge JSON
        parsed_data, is_json_valid, parse_err = self.parse_judge_json(raw_judge_text)
        if not is_json_valid:
            return self._build_fallback_result(
                comp_id=comp_id,
                run_id=actual_run_id,
                original_record_id=original_record_id,
                english_record_id=english_record_id,
                lang_code=lang_code,
                lang_name=lang_name,
                orig_text=original_answer_text,
                eng_text=english_answer_text,
                raw_judge_text=raw_judge_text,
                error_reason=parse_err or "MALFORMED_OUTPUT",
                deterministic=deterministic_summary,
                prompt_version=prompt_version,
                model_id=judge_model_id,
                model_revision=judge_revision,
                gen_config=judge_config,
                elapsed_time=elapsed_time
            )

        raw_judge_class = str(parsed_data.get("classification", "UNCERTAIN")).upper().strip()
        overall_rationale = str(parsed_data.get("overall_rationale", "")).strip()

        # Step 4: Parse Claims & Validate Evidence Substrings
        raw_claims = parsed_data.get("claims", [])
        validated_claims: List[EvidenceGroundedClaim] = []
        validation_errors: List[str] = []
        seen_claim_ids: Set[str] = set()

        for c_idx, raw_c in enumerate(raw_claims, 1):
            if not isinstance(raw_c, dict):
                validation_errors.append(f"Claim #{c_idx} is not a valid dictionary")
                continue

            c_id = str(raw_c.get("claim_id", f"CLM_{c_idx:02d}")).strip()
            if c_id in seen_claim_ids:
                validation_errors.append(f"Duplicate claim_id detected: '{c_id}'")
            seen_claim_ids.add(c_id)

            rel = str(raw_c.get("relationship", "AMBIGUOUS")).upper().strip()
            interp = str(raw_c.get("interpreted_claim", "")).strip()
            orig_span = raw_c.get("original_evidence_span")
            eng_span = raw_c.get("english_evidence_span")
            expl = str(raw_c.get("explanation", "")).strip()

            claim_obj = EvidenceGroundedClaim(
                claim_id=c_id,
                relationship=rel,
                original_evidence_span=str(orig_span) if orig_span is not None else None,
                english_evidence_span=str(eng_span) if eng_span is not None else None,
                interpreted_claim=interp,
                explanation=expl,
            )

            # Programmatically validate evidence spans
            orig_valid, eng_valid, val_err = EvidenceValidator.validate_claim(
                claim=claim_obj,
                original_text=original_answer_text,
                english_text=english_answer_text
            )

            claim_obj.is_original_evidence_valid = orig_valid
            claim_obj.is_english_evidence_valid = eng_valid
            claim_obj.validation_error = val_err

            if val_err:
                validation_errors.append(f"[{c_id}] {val_err}")

            validated_claims.append(claim_obj)

        # Determine evidence validation status
        if not validation_errors:
            evidence_status = "ALL_EVIDENCE_GROUNDED"
        elif any("NOT_FOUND_IN_SOURCE" in e for e in validation_errors):
            evidence_status = "FABRICATED_EVIDENCE_DETECTED"
        elif any("MISSING_" in e for e in validation_errors):
            evidence_status = "MISSING_REQUIRED_EVIDENCE"
        else:
            evidence_status = "SCHEMA_VIOLATION"

        # Step 5: Final Arbitration via Decision Table
        final_class, decision_src, override_reason = ClassificationArbiter.arbitrate(
            raw_judge_class=raw_judge_class,
            overall_rationale=overall_rationale,
            validated_claims=validated_claims,
            validation_status=evidence_status,
            deterministic=deterministic_summary,
        )

        return ComparisonResult(
            comparison_id=comp_id,
            run_id=actual_run_id,
            original_record_id=original_record_id,
            english_record_id=english_record_id,
            original_language_code=lang_code,
            original_language_name=lang_name,
            english_language_code="en",
            original_answer_text=original_answer_text,
            english_answer_text=english_answer_text,
            raw_judge_classification=raw_judge_class,
            final_arbiter_classification=final_class,
            classification_decision_source=decision_src,
            arbiter_override_reason=override_reason,
            overall_rationale=overall_rationale or "Factual comparison completed.",
            claims=validated_claims,
            evidence_validation_status=evidence_status,
            matched_claims=[c for c in validated_claims if c.relationship == "MATCHED"],
            contradictory_claims=[c for c in validated_claims if c.relationship == "CONTRADICTORY"],
            original_only_claims=[c for c in validated_claims if c.relationship == "ORIGINAL_ONLY"],
            english_only_claims=[c for c in validated_claims if c.relationship == "ENGLISH_ONLY"],
            ambiguous_claims=[c for c in validated_claims if c.relationship == "AMBIGUOUS"],
            deterministic_features=deterministic_summary,
            comparison_method="hybrid_deterministic_and_gemma_judge",
            prompt_version=prompt_version,
            judge_model_id=judge_model_id,
            judge_model_revision=judge_revision,
            judge_generation_config=judge_config,
            raw_judge_response=raw_judge_text,
            execution_status=execution_status,
            execution_time_seconds=elapsed_time,
        )

    @classmethod
    def _build_fallback_result(
        cls,
        comp_id: str,
        run_id: str,
        original_record_id: Optional[str],
        english_record_id: Optional[str],
        lang_code: str,
        lang_name: str,
        orig_text: str,
        eng_text: str,
        raw_judge_text: str,
        error_reason: str,
        deterministic: DeterministicFeatureSummary,
        prompt_version: str,
        model_id: str,
        model_revision: str,
        gen_config: Dict[str, Any],
        elapsed_time: float
    ) -> ComparisonResult:
        return ComparisonResult(
            comparison_id=comp_id,
            run_id=run_id,
            original_record_id=original_record_id,
            english_record_id=english_record_id,
            original_language_code=lang_code,
            original_language_name=lang_name,
            english_language_code="en",
            original_answer_text=orig_text,
            english_answer_text=eng_text,
            raw_judge_classification="MALFORMED",
            final_arbiter_classification="UNCERTAIN",
            classification_decision_source="MALFORMED_OUTPUT_FALLBACK",
            arbiter_override_reason=f"Failed to parse judge output: {error_reason}",
            overall_rationale=f"Comparison uncertain due to judge failure: {error_reason}",
            claims=[],
            evidence_validation_status="PARSING_FAILED",
            matched_claims=[],
            contradictory_claims=[],
            original_only_claims=[],
            english_only_claims=[],
            ambiguous_claims=[],
            deterministic_features=deterministic,
            comparison_method="hybrid_deterministic_and_gemma_judge",
            prompt_version=prompt_version,
            judge_model_id=model_id,
            judge_model_revision=model_revision,
            judge_generation_config=gen_config,
            raw_judge_response=raw_judge_text,
            execution_status=f"PARSING_ERROR: {error_reason}",
            execution_time_seconds=elapsed_time,
        )
