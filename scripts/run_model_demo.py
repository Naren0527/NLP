"""
Live Model Demo: Gemma 3 4B Instruct on local RTX 4050 GPU.
Executes the shortest genuine inference path, displaying:
- Input Question
- Model Answer
- Statistical Confidence (geometric mean token probability)
- Verbalized Confidence (strict regex percentage elicitation)
- Language / Condition
Saves the demo output to outputs/urgent_results/model_demo.txt
"""

import json
import os
import sys
import io
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set line buffering for UTF-8 Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_engine import ModelEngine
from src.confidence import ConfidenceEngine

DEMO_OUTPUT_FILE = Path("outputs/urgent_results/model_demo.txt")
DEMO_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def run_demo():
    print("=" * 70, flush=True)
    print("  LIVE EMPIRICAL MODEL DEMO: GEMMA 3 4B INSTRUCT (RTX 4050 GPU)", flush=True)
    print("=" * 70, flush=True)

    demo_lines = []
    def log_print(text):
        print(text, flush=True)
        demo_lines.append(text)

    log_print("=" * 70)
    log_print("  LIVE EMPIRICAL MODEL DEMO: GEMMA 3 4B INSTRUCT (RTX 4050 GPU)")
    log_print("=" * 70)

    # 1. Initialize Engines
    t0 = time.time()
    model_engine = ModelEngine.get_instance()
    conf_engine = ConfidenceEngine(model_engine=model_engine)

    # Ensure model is loaded on GPU
    if not model_engine.is_loaded():
        log_print("Loading Gemma 3 4B Instruct with 4-bit NF4 quantization onto CUDA...")
        model_engine.load_model()
    log_print(f"Engine Ready. Device: {model_engine.device} | Revision: {model_engine.model_revision[:12]}...")

    # Demo Cases: 1 English, 1 Hindi
    test_cases = [
        {
            "condition": "en",
            "language": "English",
            "question": "What is the capital of India and which river flows through it?",
            "prompt": "You are answering a user question. The user's language is English. Answer only in English. Preserve names, numbers, dates and negation. Question: What is the capital of India and which river flows through it?",
            "answer_language": "en",
        },
        {
            "condition": "hi",
            "language": "Hindi",
            "question": "भारत की राजधानी क्या है और इसके पास कौन सी मुख्य नदी बहती है?",
            "prompt": "You are answering a user question. The user's language is Hindi. Answer only in Hindi. Preserve names, numbers, dates and negation. Question: भारत की राजधानी क्या है और इसके पास कौन सी मुख्य नदी बहती है?",
            "answer_language": "hi",
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        log_print("-" * 70)
        log_print(f"TEST CASE {idx}: [{case['language'].upper()} - Condition '{case['condition']}']")
        log_print(f"INPUT QUESTION:         {case['question']}")
        log_print(f"LANGUAGE / CONDITION:   {case['language']} ({case['condition']})")

        # 1. Answer Generation
        t_gen = time.time()
        gen_out = model_engine.generate(prompt=case["prompt"], max_new_tokens=64, apply_template=True)
        elapsed_gen = time.time() - t_gen

        # 2. Verbalized Confidence Elicitation
        t_conf = time.time()
        conf_res = conf_engine.elicit_verbalized_confidence(
            question=case["question"],
            answer=gen_out.answer_text,
            answer_language=case["answer_language"],
        )
        elapsed_conf = time.time() - t_conf

        stat_conf = gen_out.statistical_confidence.content_tokens.geometric_probability if gen_out.statistical_confidence else gen_out.geometric_probability

        log_print(f"\nMODEL ANSWER:\n{gen_out.answer_text.strip()}")
        log_print(f"\nSTATISTICAL CONFIDENCE: {stat_conf:.4f} (Content Tokens Geometric Probability)")
        log_print(f"VERBALIZED CONFIDENCE:  {conf_res.confidence_score} ({conf_res.raw_response.strip()}) [Valid: {conf_res.is_valid}]")
        log_print(f"INFERENCE LATENCY:      Generation: {elapsed_gen:.2f}s | Confidence: {elapsed_conf:.2f}s")
        log_print(f"TOKENS GENERATED:       {gen_out.generated_token_count} tokens")

    log_print("=" * 70)
    log_print(f"DEMO COMPLETED SUCCESSFULLY IN {time.time() - t0:.2f}s")
    log_print("=" * 70)

    with open(DEMO_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(demo_lines) + "\n")

    print(f"\nDemo output successfully saved to: {DEMO_OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    run_demo()
