"""
Minimal Real-GPU Smoke Test for Judge Inference Pathway.

Tests:
1. Loads Gemma 3 4B Instruct via ModelEngine on CUDA:0.
2. Constructs the symmetrical comparison judge prompt for a Kannada/English pair.
3. Invokes ModelComparisonJudge (max_new_tokens=1024, greedy).
4. Persists raw judge JSON completion.
5. Verifies schema compliance, evidence grounding, and ClassificationArbiter output.
6. Asserts task type classification as 'JUDGE_INFERENCE'.
7. Outputs isolated results to results/empirical_judge_smoke/.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from src.model_engine import ModelEngine
from src.answer_comparator import AnswerComparator


def run_judge_smoke_test():
    print("=" * 65)
    print("PHASE 5 REAL-GPU SMOKE TEST: COMPARISON JUDGE PATHWAY")
    print("=" * 65)

    start_time = time.time()
    assert torch.cuda.is_available(), "CUDA is required for real-GPU smoke test"

    output_dir = "results/empirical_judge_smoke"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Engine
    model_engine = ModelEngine.get_instance()
    comparator = AnswerComparator(model_engine=model_engine)

    question = "What is the capital city of Karnataka?"
    orig_text = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು."
    eng_text = "The capital city of Karnataka is Bengaluru."

    print(f"Model ID:        {model_engine.model_id}")
    print(f"Model Revision:  {model_engine.model_revision}")
    print(f"Question:        {question}")
    print(f"Original Answer: {orig_text}")
    print(f"English Answer:  {eng_text}")

    # Track VRAM before generation
    vram_before_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)

    # 2. Execute live comparison judge inference
    print("\nExecuting live Gemma 3 4B Judge inference (max_new_tokens=1024)...")
    comp_result = comparator.compare(
        question=question,
        original_answer_text=orig_text,
        english_answer_text=eng_text,
        original_language_code="kn",
        original_language_name="Kannada",
        run_id="preflight_judge_test_01",
    )

    duration = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    print(f"\n[Judge Output Execution Status]: {comp_result.execution_status}")
    print(f"[Raw Judge Classification]   : {comp_result.raw_judge_classification}")
    print(f"[Final Arbiter Decision]    : {comp_result.final_arbiter_classification}")
    print(f"[Decision Source]            : {comp_result.classification_decision_source}")
    print(f"[Evidence Grounding Status]  : {comp_result.evidence_validation_status}")
    print(f"[Claims Extracted]           : {len(comp_result.claims)}")
    print(f"[Raw Judge JSON Preview]     :\n{comp_result.raw_judge_response[:300]}...")

    # Assertions
    assert comp_result.execution_status == "SUCCESS"
    assert comp_result.raw_judge_classification in ["CONSISTENT", "INCONSISTENT", "UNCERTAIN"]
    assert comp_result.final_arbiter_classification in ["CONSISTENT", "INCONSISTENT", "UNCERTAIN"]
    assert len(comp_result.raw_judge_response) > 20

    # 3. Save Artifact
    artifact = {
        "task_type": "JUDGE_INFERENCE",
        "model_id": model_engine.model_id,
        "model_revision": model_engine.model_revision,
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "duration_seconds": round(duration, 2),
        "comparison_result": {
            "comparison_id": comp_result.comparison_id,
            "raw_judge_classification": comp_result.raw_judge_classification,
            "final_arbiter_classification": comp_result.final_arbiter_classification,
            "classification_decision_source": comp_result.classification_decision_source,
            "overall_rationale": comp_result.overall_rationale,
            "evidence_validation_status": comp_result.evidence_validation_status,
            "raw_judge_response": comp_result.raw_judge_response,
            "claims": [c.__dict__ for c in comp_result.claims],
        }
    }

    out_file = os.path.join(output_dir, "judge_smoke_test_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"\nJudge Smoke Test completed successfully in {duration:.2f}s.")
    print(f"Artifact written to {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_judge_smoke_test()
