"""
Phase 5 Eligibility Filters and Two-Tier Attrition Accounting.

Implements Section 9 of Phase 5 Methodological Specification v4.5:
- Record-level eligibility indicators: E_{l, strict}(q), E_{l, verb}(q), E_{l, stat}(q).
- Exact record-level intersections for paired cohorts:
  * Target Condition l vs English reference:
    N_paired_verb(l, P) = sum_{q in P} (E_{en,strict} * E_{en,verb} * E_{l,strict} * E_{l,verb})
    N_paired_stat(l, P) = sum_{q in P} (E_{en,strict} * E_{en,stat} * E_{l,strict} * E_{l,stat})
  * English Self-Paired:
    N_paired_verb(en, P) = sum_{q in P} (E_{en,strict} * E_{en,verb})
    N_paired_stat(en, P) = sum_{q in P} (E_{en,strict} * E_{en,stat})
- Table A (Full Dataset Accounting) and Table B (Partition-Specific Paired Cohort Accounting).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass
class RecordEligibility:
    question_id: str
    condition: str
    strict_correctness_eligible: bool
    verbalized_valid: bool
    statistical_valid: bool

    @property
    def dual_track_eligible(self) -> bool:
        return self.strict_correctness_eligible and self.verbalized_valid and self.statistical_valid

    @property
    def verbalized_track_eligible(self) -> bool:
        return self.strict_correctness_eligible and self.verbalized_valid

    @property
    def statistical_track_eligible(self) -> bool:
        return self.strict_correctness_eligible and self.statistical_valid


@dataclass
class TableARow:
    condition: str
    total_generated: int
    strict_correctness_eligible: int
    verbalized_valid: int
    statistical_valid: int
    dual_track_eligible: int
    attrition_rate_percent: float


@dataclass
class TableBRow:
    partition_name: str
    condition: str
    partition_size: int
    max_paired_cohort: int
    actual_paired_verbalized: int
    actual_paired_statistical: int
    attrition_rate_percent_verb: float
    attrition_rate_percent_stat: float


@dataclass
class AttritionSummary:
    table_a: List[TableARow]
    table_b: List[TableBRow]
    total_generated: int
    total_dual_eligible: int


class EligibilityEngine:
    """
    Computes eligibility indicators and exact intersection paired cohorts.
    """

    def __init__(self, records: List[Dict[str, Any]]):
        """
        records: list of dictionaries representing generated evaluation records.
        Expected keys per record:
          - question_id
          - condition
          - partition (cal / val / test)
          - is_strict_correctness_eligible (bool)
          - is_verbalized_valid (bool)
          - is_statistical_valid (bool)
        """
        self.records = records
        # Map: (condition, question_id) -> RecordEligibility
        self.eligibility_map: Dict[Tuple[str, str], RecordEligibility] = {}
        # Map: question_id -> partition
        self.question_partitions: Dict[str, str] = {}
        # Set of all conditions
        self.conditions: Set[str] = set()

        for r in records:
            qid = str(r["question_id"])
            cond = str(r["condition"])
            part = str(r.get("partition", "unknown"))
            self.question_partitions[qid] = part
            self.conditions.add(cond)

            strict_ok = bool(r.get("is_strict_correctness_eligible", False))
            verb_ok = bool(r.get("is_verbalized_valid", False))
            stat_ok = bool(r.get("is_statistical_valid", False))

            self.eligibility_map[(cond, qid)] = RecordEligibility(
                question_id=qid,
                condition=cond,
                strict_correctness_eligible=strict_ok,
                verbalized_valid=verb_ok,
                statistical_valid=stat_ok,
            )

    def get_paired_cohort_verbalized(
        self,
        target_condition: str,
        question_subset: List[str],
        reference_condition: str = "en",
    ) -> List[str]:
        """
        Computes exact record-level intersection for verbalized track:
        If target == reference ('en'):
          E_{en,strict}(q) * E_{en,verb}(q)
        Else:
          E_{ref,strict}(q) * E_{ref,verb}(q) * E_{target,strict}(q) * E_{target,verb}(q)
        """
        paired_qids: List[str] = []
        for q in question_subset:
            ref_elig = self.eligibility_map.get((reference_condition, q))
            if not ref_elig or not ref_elig.verbalized_track_eligible:
                continue

            if target_condition == reference_condition:
                paired_qids.append(q)
            else:
                target_elig = self.eligibility_map.get((target_condition, q))
                if target_elig and target_elig.verbalized_track_eligible:
                    paired_qids.append(q)

        return paired_qids

    def get_paired_cohort_statistical(
        self,
        target_condition: str,
        question_subset: List[str],
        reference_condition: str = "en",
    ) -> List[str]:
        """
        Computes exact record-level intersection for statistical track:
        If target == reference ('en'):
          E_{en,strict}(q) * E_{en,stat}(q)
        Else:
          E_{ref,strict}(q) * E_{ref,stat}(q) * E_{target,strict}(q) * E_{target,stat}(q)
        """
        paired_qids: List[str] = []
        for q in question_subset:
            ref_elig = self.eligibility_map.get((reference_condition, q))
            if not ref_elig or not ref_elig.statistical_track_eligible:
                continue

            if target_condition == reference_condition:
                paired_qids.append(q)
            else:
                target_elig = self.eligibility_map.get((target_condition, q))
                if target_elig and target_elig.statistical_track_eligible:
                    paired_qids.append(q)

        return paired_qids

    def compute_table_a(self, all_question_ids: List[str]) -> List[TableARow]:
        """Computes Table A: Full Dataset Attrition Accounting."""
        rows: List[TableARow] = []
        sorted_conditions = sorted(list(self.conditions))
        total_q = len(all_question_ids)

        for cond in sorted_conditions:
            gen_count = 0
            strict_ok_cnt = 0
            verb_ok_cnt = 0
            stat_ok_cnt = 0
            dual_ok_cnt = 0

            for q in all_question_ids:
                elig = self.eligibility_map.get((cond, q))
                if elig:
                    gen_count += 1
                    if elig.strict_correctness_eligible:
                        strict_ok_cnt += 1
                    if elig.verbalized_valid:
                        verb_ok_cnt += 1
                    if elig.statistical_valid:
                        stat_ok_cnt += 1
                    if elig.dual_track_eligible:
                        dual_ok_cnt += 1

            attrition_pct = 100.0 * (1.0 - (dual_ok_cnt / gen_count)) if gen_count > 0 else 100.0

            rows.append(
                TableARow(
                    condition=cond,
                    total_generated=gen_count,
                    strict_correctness_eligible=strict_ok_cnt,
                    verbalized_valid=verb_ok_cnt,
                    statistical_valid=stat_ok_cnt,
                    dual_track_eligible=dual_ok_cnt,
                    attrition_rate_percent=round(attrition_pct, 2),
                )
            )
        return rows

    def compute_table_b(
        self,
        partition_name: str,
        partition_qids: List[str],
        reference_condition: str = "en",
    ) -> List[TableBRow]:
        """Computes Table B: Partition-Specific Paired Cohort Accounting."""
        rows: List[TableBRow] = []
        sorted_conditions = sorted(list(self.conditions))
        part_size = len(partition_qids)

        for cond in sorted_conditions:
            paired_verb = self.get_paired_cohort_verbalized(cond, partition_qids, reference_condition)
            paired_stat = self.get_paired_cohort_statistical(cond, partition_qids, reference_condition)

            n_verb = len(paired_verb)
            n_stat = len(paired_stat)

            attr_verb = 100.0 * (1.0 - (n_verb / part_size)) if part_size > 0 else 100.0
            attr_stat = 100.0 * (1.0 - (n_stat / part_size)) if part_size > 0 else 100.0

            rows.append(
                TableBRow(
                    partition_name=partition_name,
                    condition=cond,
                    partition_size=part_size,
                    max_paired_cohort=part_size,
                    actual_paired_verbalized=n_verb,
                    actual_paired_statistical=n_stat,
                    attrition_rate_percent_verb=round(attr_verb, 2),
                    attrition_rate_percent_stat=round(attr_stat, 2),
                )
            )
        return rows
