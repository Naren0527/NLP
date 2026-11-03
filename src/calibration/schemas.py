"""
Phase 5 Claim-Level Verification and Answer Correctness Schemas.

Implements Section 4 of Phase 5 Methodological Specification v4.5:
- 3-State Claim Rubric: SUPPORTED (1), REFUTED (0), UNCERTAIN (0).
- Answer-Level Binary Correctness Formulas:
  * Strict: Product over all claims (all must be SUPPORTED).
  * Fractional: Mean proportion of SUPPORTED claims.
  * Lenient: At least one SUPPORTED and zero REFUTED.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class ClaimState(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ClaimVerification:
    claim_id: str
    proposition: str
    state: ClaimState
    evidence_span: Optional[str] = None
    evidence_tier: int = 1  # Tier 1: Authoritative, Tier 2: Scholarly, Tier 3: Public
    adjudicator_id: Optional[str] = None
    notes: Optional[str] = None

    @property
    def binary_score(self) -> int:
        """Conservative epistemic score: only SUPPORTED is 1, REFUTED and UNCERTAIN are 0."""
        return 1 if self.state == ClaimState.SUPPORTED else 0


@dataclass
class AnswerVerification:
    answer_id: str
    claims: List[ClaimVerification] = field(default_factory=list)
    is_ambiguous: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.state == ClaimState.SUPPORTED)

    @property
    def refuted_count(self) -> int:
        return sum(1 for c in self.claims if c.state == ClaimState.REFUTED)

    @property
    def uncertain_count(self) -> int:
        return sum(1 for c in self.claims if c.state == ClaimState.UNCERTAIN)

    @property
    def strict_correctness(self) -> int:
        """
        Strict Correctness: Y_i^strict = prod_{k=1}^{K_i} I[c_{i,k} == 'SUPPORTED']
        Empty claims list defaults to 0 (unsupported).
        """
        if not self.claims:
            return 0
        return 1 if (self.supported_count == self.claim_count and self.refuted_count == 0 and self.uncertain_count == 0) else 0

    @property
    def fractional_correctness(self) -> float:
        """
        Partial / Fractional Correctness: (1 / K_i) * sum I[c_{i,k} == 'SUPPORTED']
        """
        if not self.claims:
            return 0.0
        return float(self.supported_count) / float(self.claim_count)

    @property
    def lenient_correctness(self) -> int:
        """
        Lenient Correctness: At least 1 SUPPORTED and 0 REFUTED.
        """
        if not self.claims:
            return 0
        return 1 if (self.supported_count >= 1 and self.refuted_count == 0) else 0


class CorrectnessEvaluator:
    """Helper evaluator for bulk verification and metrics generation."""

    @staticmethod
    def evaluate_answer(
        answer_id: str,
        claim_records: List[Dict[str, Any]],
        is_ambiguous: bool = False,
    ) -> AnswerVerification:
        verifications: List[ClaimVerification] = []
        for cr in claim_records:
            state_str = str(cr.get("state", "UNCERTAIN")).upper()
            state = ClaimState[state_str] if state_str in ClaimState.__members__ else ClaimState.UNCERTAIN
            verifications.append(
                ClaimVerification(
                    claim_id=cr.get("claim_id", f"CLM_{len(verifications)+1:02d}"),
                    proposition=cr.get("proposition", ""),
                    state=state,
                    evidence_span=cr.get("evidence_span"),
                    evidence_tier=int(cr.get("evidence_tier", 1)),
                    adjudicator_id=cr.get("adjudicator_id"),
                    notes=cr.get("notes"),
                )
            )
        return AnswerVerification(
            answer_id=answer_id,
            claims=verifications,
            is_ambiguous=is_ambiguous,
        )
