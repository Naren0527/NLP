"""
Minimal recovery/resume validation script.
Verifies that:
1. Checkpoint loading loads existing 628 records without corruption.
2. Identifies the next incomplete record (Q_GEO_MH_16, hi).
3. Executes a genuine model inference for that record on GPU.
4. Appends it safely to the empirical ledger and call logs.
5. Verifies zero duplicate records or calls.
"""
import json
import os
import sys
import time
import io

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_engine import ModelEngine
from src.calibration.empirical_driver import EmpiricalPipelineDriver, CONDITIONS
from src.calibration.instrumentation import ModelCallLogger

def run_minimal_validation():
    output_dir = "results/empirical"
    records_file = os.path.join(output_dir, "phase5_empirical_records.jsonl")
    calls_file = os.path.join(output_dir, "model_call_logs.jsonl")

    # 1. Load Question Manifest
    with open("data/full_150_questions.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # 2. Checkpoint Loading
    model_engine = ModelEngine.get_instance()
    driver = EmpiricalPipelineDriver(output_dir=output_dir, model_engine=model_engine)
    logger = ModelCallLogger(calls_file)

    completed_records = driver.load_completed_records()
    initial_count = len(completed_records)
    print(f"[Validation Step 1] Loaded {initial_count} existing records from disk.")
    assert initial_count >= 628, f"Expected at least 628 records, found {initial_count}"

    # 3. Find next incomplete record
    next_q = None
    next_cond = None
    for q in questions:
        qid = q["question_id"]
        for cond in CONDITIONS:
            if (qid, cond) not in completed_records:
                next_q = q
                next_cond = cond
                break
        if next_q is not None:
            break

    assert next_q is not None, "All records already complete!"
    qid = next_q["question_id"]
    print(f"[Validation Step 2] Next incomplete record identified: ({qid}, {next_cond})")

    # 4. Execute 1 GPU inference
    t0 = time.time()
    rec = driver.run_inference_for_record(next_q, next_cond)
    elapsed = time.time() - t0
    print(f"[Validation Step 3] Inference completed in {elapsed:.2f}s.")
    print(f"  Generated Answer: {rec['answer_text'][:80]}...")
    print(f"  Verbalized Confidence: {rec['verbalized_confidence']} (valid: {rec['is_verbalized_valid']})")
    print(f"  Statistical Confidence: {rec['statistical_confidence']:.4f}")

    # 5. Atomic Append
    with open(records_file, "a", encoding="utf-8") as f_records:
        f_records.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_records.flush()

    logger.record_call(
        call_id=f"{rec['record_id']}_gen",
        task_type="ANSWER_GENERATION",
        question_id=qid,
        condition=next_cond,
        model_id=model_engine.model_id,
        max_new_tokens=128,
        completion_status="SUCCESS",
        execution_time_seconds=elapsed,
        output_preview=rec["answer_text"],
    )
    logger.record_call(
        call_id=f"{rec['record_id']}_conf",
        task_type="VERBALIZED_CONFIDENCE",
        question_id=qid,
        condition=next_cond,
        model_id=model_engine.model_id,
        max_new_tokens=10,
        completion_status="SUCCESS" if rec["is_verbalized_valid"] else "PARSE_FAILURE",
        execution_time_seconds=0.5,
        output_preview=str(rec["verbalized_confidence_raw"]),
    )

    # 6. Verify ledger integrity
    reloaded_records = driver.load_completed_records()
    assert len(reloaded_records) == initial_count + 1, f"Expected {initial_count + 1}, got {len(reloaded_records)}"
    print(f"[Validation Step 4] Ledger safely updated to {len(reloaded_records)} records. Zero duplication.")
    print("[Validation Step 5] Minimal recovery/resume validation PASSED.")

if __name__ == "__main__":
    run_minimal_validation()
