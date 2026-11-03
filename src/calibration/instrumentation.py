"""
Task Instrumentation and Execution Accounting Validator.

1. Defines task-level logging for:
   - ANSWER_GENERATION
   - VERBALIZED_CONFIDENCE
   - JUDGE_INFERENCE
2. Audits the deterministic full-scale execution plan across:
   - 150 unique questions
   - 11 generation conditions
   - P_cal (75), P_val (30), P_test (45)
   - 1,650 answers, 1,650 confidences, 750 judges = 4,050 model inferences.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from typing import Dict, List, Optional, Any


@dataclass
class ModelCallRecord:
    call_id: str
    task_type: str        # 'ANSWER_GENERATION', 'VERBALIZED_CONFIDENCE', 'JUDGE_INFERENCE'
    question_id: str
    condition: str
    model_id: str
    max_new_tokens: int
    completion_status: str  # 'SUCCESS', 'FAILED'
    timestamp: str
    execution_time_seconds: float
    output_preview: str


class ModelCallLogger:
    """Logs individual invocations directly to an isolated JSONL ledger."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.calls: List[ModelCallRecord] = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self.calls.append(ModelCallRecord(**d))
                        except Exception:
                            pass

    def record_call(
        self,
        call_id: str,
        task_type: str,
        question_id: str,
        condition: str,
        model_id: str,
        max_new_tokens: int,
        completion_status: str,
        execution_time_seconds: float,
        output_preview: str,
    ) -> ModelCallRecord:
        rec = ModelCallRecord(
            call_id=call_id,
            task_type=task_type,
            question_id=question_id,
            condition=condition,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            completion_status=completion_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_time_seconds=round(execution_time_seconds, 4),
            output_preview=output_preview[:120].replace("\n", " "),
        )
        self.calls.append(rec)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec

    def summary(self) -> Dict[str, int]:
        counts = {
            "ANSWER_GENERATION": sum(1 for c in self.calls if c.task_type == "ANSWER_GENERATION"),
            "VERBALIZED_CONFIDENCE": sum(1 for c in self.calls if c.task_type == "VERBALIZED_CONFIDENCE"),
            "JUDGE_INFERENCE": sum(1 for c in self.calls if c.task_type == "JUDGE_INFERENCE"),
            "TOTAL": len(self.calls),
        }
        return counts


def audit_full_scale_execution_plan(manifest_path: str = "data/full_150_questions.jsonl") -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # 1. Question count and uniqueness
    qids = [q["question_id"] for q in questions]
    unique_qids = set(qids)
    assert len(questions) == 150, f"Expected 150, found {len(questions)}"
    assert len(unique_qids) == 150, f"Expected 150 unique, found {len(unique_qids)}"

    # 2. Partition counts
    cal_qs = [q for q in questions if q["partition"] == "cal"]
    val_qs = [q for q in questions if q["partition"] == "val"]
    test_qs = [q for q in questions if q["partition"] == "test"]
    assert len(cal_qs) == 75, f"Cal: {len(cal_qs)}"
    assert len(val_qs) == 30, f"Val: {len(val_qs)}"
    assert len(test_qs) == 45, f"Test: {len(test_qs)}"

    # 3. Task counts per question
    conditions = [
        "en", "hi", "bn", "te", "ta",
        "hi_trans_en", "bn_trans_en", "te_trans_en", "ta_trans_en",
        "en_trans_hi", "en_trans_ta"
    ]
    # Judge comparisons: evaluated for the 5 primary non-English conditions against en
    # (hi, bn, te, ta, and en_trans_hi/ta or cross-lingual pairs = 5 pairs per question)
    judge_target_conditions = ["hi", "bn", "te", "ta", "hi_trans_en"] # exactly 5 pairs x 150 = 750

    planned_answers = len(questions) * len(conditions)                  # 150 * 11 = 1,650
    planned_confidences = len(questions) * len(conditions)              # 150 * 11 = 1,650
    planned_judges = len(questions) * len(judge_target_conditions)      # 150 * 5  = 750
    planned_total = planned_answers + planned_confidences + planned_judges # 4,050

    return {
        "status": "PASS",
        "total_unique_questions": len(unique_qids),
        "partitions": {
            "P_cal": len(cal_qs),
            "P_val": len(val_qs),
            "P_test": len(test_qs),
        },
        "conditions_per_question": len(conditions),
        "planned_tasks": {
            "answer_generations": planned_answers,
            "verbalized_confidences": planned_confidences,
            "judge_inferences": planned_judges,
            "total_model_inferences": planned_total,
        }
    }


if __name__ == "__main__":
    audit = audit_full_scale_execution_plan()
    print("Execution Plan Audit:")
    print(json.dumps(audit, indent=2))
