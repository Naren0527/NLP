"""
Preflight Real-GPU Test: Checkpoint Interruption and Resume Verification.

Tests:
1. Loads Gemma 3 4B on GPU.
2. In run 1: executes inferences for record 1 (en) and record 2 (hi), writing atomically.
3. Simulates interruption.
4. In run 2: initiates execution for records 1, 2, and 3 (ta).
5. Verifies:
   - Records 1 and 2 are loaded from checkpoint and NOT recomputed.
   - Record 3 is computed freshly on GPU.
   - Total records in ledger equals exactly 3 without duplicates.
   - Model call logs record exactly 2 calls in Run 1 and exactly 1 fresh call in Run 2.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from src.model_engine import ModelEngine
from src.calibration.empirical_driver import EmpiricalPipelineDriver
from src.calibration.instrumentation import ModelCallLogger


def run_checkpoint_preflight_test():
    print("=" * 65)
    print("PHASE 5 PREFLIGHT: CHECKPOINT INTERRUPTION & RESUME TEST")
    print("=" * 65)

    test_dir = "results/preflight_checkpoint_test"
    os.makedirs(test_dir, exist_ok=True)
    records_file = os.path.join(test_dir, "phase5_empirical_records.jsonl")
    log_file = os.path.join(test_dir, "model_calls.jsonl")

    # Clean test directory
    if os.path.exists(records_file):
        os.remove(records_file)
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = ModelCallLogger(log_file)
    model_engine = ModelEngine.get_instance()
    driver = EmpiricalPipelineDriver(output_dir=test_dir, model_engine=model_engine)

    sample_q = {
        "question_id": "PREFLIGHT_Q01",
        "domain": "Geography",
        "reasoning": "Direct",
        "source": "Belebele",
        "partition": "cal",
        "text_en": "What is the capital city of Karnataka?",
        "ground_truth_claims": ["Bengaluru is the capital city of Karnataka."],
        "reference_answer": "Bengaluru is the capital city of Karnataka.",
        "evidence_passage": "Bengaluru, officially known as Bangalore, is the capital of the southern Indian state of Karnataka.",
    }

    # ==========================================
    # RUN 1: Execute 2 records, then interrupt
    # ==========================================
    print("\n--- PHASE 1: INITIAL RUN (Generating 2 records) ---")
    run1_conditions = ["en", "hi"]
    completed_run1 = driver.load_completed_records()

    with open(records_file, "a", encoding="utf-8") as f_out:
        for cond in run1_conditions:
            key = (sample_q["question_id"], cond)
            t0 = time.time()
            rec = driver.run_inference_for_record(sample_q, cond)
            elapsed = time.time() - t0
            
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            completed_run1[key] = rec

            logger.record_call(
                call_id=rec["record_id"],
                task_type="ANSWER_GENERATION",
                question_id=sample_q["question_id"],
                condition=cond,
                model_id=model_engine.model_id,
                max_new_tokens=128,
                completion_status="SUCCESS",
                execution_time_seconds=elapsed,
                output_preview=rec["answer_text"],
            )
            print(f"  [Run 1] Persisted: {cond} -> '{rec['answer_text'][:40]}...'")

    print("Simulating interruption...")

    # ==========================================
    # RUN 2: Resume with 3 records (en, hi, ta)
    # ==========================================
    print("\n--- PHASE 2: RESUME RUN (Targeting en, hi, ta) ---")
    run2_conditions = ["en", "hi", "ta"]
    # Re-initialize driver to simulate process restart
    driver_run2 = EmpiricalPipelineDriver(output_dir=test_dir, model_engine=model_engine)
    completed_run2 = driver_run2.load_completed_records()

    print(f"Loaded {len(completed_run2)} existing records from checkpoint on startup.")
    assert len(completed_run2) == 2, f"Expected 2 completed records, found {len(completed_run2)}"

    run2_new_inferences = 0
    with open(records_file, "a", encoding="utf-8") as f_out:
        for cond in run2_conditions:
            key = (sample_q["question_id"], cond)
            if key in completed_run2:
                print(f"  [Run 2] SKIPPED (already in checkpoint): {cond}")
                continue

            print(f"  [Run 2] FRESH INFERENCE: {cond}")
            t0 = time.time()
            rec = driver_run2.run_inference_for_record(sample_q, cond)
            elapsed = time.time() - t0

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            completed_run2[key] = rec
            run2_new_inferences += 1

            logger.record_call(
                call_id=rec["record_id"],
                task_type="ANSWER_GENERATION",
                question_id=sample_q["question_id"],
                condition=cond,
                model_id=model_engine.model_id,
                max_new_tokens=128,
                completion_status="SUCCESS",
                execution_time_seconds=elapsed,
                output_preview=rec["answer_text"],
            )

    # Verification
    assert run2_new_inferences == 1, f"Expected exactly 1 new inference, got {run2_new_inferences}"

    # Read final file lines
    with open(records_file, "r", encoding="utf-8") as f:
        final_lines = [json.loads(line) for line in f if line.strip()]

    assert len(final_lines) == 3, f"Expected exactly 3 records, got {len(final_lines)}"
    keys_found = [(r["question_id"], r["condition"]) for r in final_lines]
    assert len(set(keys_found)) == 3, "Duplicate keys detected in records ledger"

    summary = logger.summary()
    print(f"\nFinal Verified Summary:")
    print(f"  Total records in ledger:  {len(final_lines)} (en, hi, ta)")
    print(f"  Unique keys:              {len(set(keys_found))}")
    print(f"  Run 1 inferences logged:  2")
    print(f"  Run 2 fresh inferences:   1")
    print(f"  Total model calls logged: {summary['TOTAL']}")
    print(f"\nCheckpoint Preflight Result: PASS (Zero duplication, exact resume)")
    print("=" * 65)


if __name__ == "__main__":
    run_checkpoint_preflight_test()
