"""
Explicit Real-GPU Smoke Test for Empirical Execution Pipeline.

Tests:
- Loads Google Gemma 3 4B Instruct model in 4-bit NF4 quantization on CUDA:0.
- Runs 2 actual question-condition inferences through EmpiricalPipelineDriver.
- Records actual model ID, model revision, allocated VRAM, token logprobs,
  verbalized confidence, and execution duration.
- Outputs isolated test artifact in results/empirical/smoke_test_output.json.
"""

import json
import os
import sys
import io
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model_engine import ModelEngine
from src.calibration.empirical_driver import EmpiricalPipelineDriver


def run_gpu_smoke_test():
    print("=" * 60)
    print("PHASE 5 REAL-GPU SMOKE TEST")
    print("=" * 60)

    start_time = time.time()
    assert torch.cuda.is_available(), "CUDA must be available for GPU smoke test"

    output_dir = "results/empirical_smoke"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize ModelEngine (Loads Gemma 3 4B in NF4)
    model_engine = ModelEngine.get_instance()
    device_name = torch.cuda.get_device_name(0)
    vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)

    print(f"Loaded Model ID:   {model_engine.model_id}")
    print(f"Model Revision:   {model_engine.model_revision}")
    print(f"GPU Device:       {device_name}")
    print(f"Allocated VRAM:   {vram_alloc_mb:.2f} MB")

    driver = EmpiricalPipelineDriver(
        output_dir=output_dir,
        model_engine=model_engine,
    )

    # Sample test question
    sample_q = {
        "question_id": "SMOKE_Q01",
        "domain": "Geography",
        "reasoning": "Direct",
        "source": "Belebele",
        "partition": "cal",
        "text_en": "What is the capital city of Karnataka?",
        "ground_truth_claims": ["Bengaluru is the capital city of Karnataka."],
        "reference_answer": "Bengaluru is the capital city of Karnataka.",
        "evidence_passage": "Bengaluru, officially known as Bangalore, is the capital of the southern Indian state of Karnataka.",
    }

    # Test Condition 1: English (en)
    print("\nExecuting real GPU inference 1: 'en' condition...")
    rec_en = driver.run_inference_for_record(sample_q, "en")
    print(f"  Answer: '{rec_en['answer_text']}'")
    print(f"  Verbalized: '{rec_en['verbalized_confidence_raw']}' (score={rec_en['verbalized_confidence']})")
    print(f"  Statistical: {rec_en['statistical_confidence']:.4f}")
    print(f"  Tokens: {rec_en['token_count']}, Logprobs length: {len(rec_en['token_logprobs'])}")

    # Test Condition 2: Hindi (hi)
    print("\nExecuting real GPU inference 2: 'hi' condition...")
    rec_hi = driver.run_inference_for_record(sample_q, "hi")
    print(f"  Answer: '{rec_hi['answer_text']}'")
    print(f"  Verbalized: '{rec_hi['verbalized_confidence_raw']}' (score={rec_hi['verbalized_confidence']})")
    print(f"  Statistical: {rec_hi['statistical_confidence']:.4f}")
    print(f"  Tokens: {rec_hi['token_count']}, Logprobs length: {len(rec_hi['token_logprobs'])}")

    duration = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    smoke_summary = {
        "status": "PASS",
        "model_id": model_engine.model_id,
        "model_revision": model_engine.model_revision,
        "device": device_name,
        "allocated_vram_mb": round(vram_alloc_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "inferences_executed": 4, # 2 answers + 2 confidences
        "duration_seconds": round(duration, 2),
        "records": [rec_en, rec_hi],
    }

    smoke_path = os.path.join(output_dir, "smoke_test_summary.json")
    with open(smoke_path, "w", encoding="utf-8") as f:
        json.dump(smoke_summary, f, indent=2, ensure_ascii=False)

    print(f"\nReal-GPU Smoke Test completed in {duration:.2f}s. Peak VRAM: {peak_vram_mb:.2f} MB.")
    print(f"Summary written to {smoke_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_gpu_smoke_test()
