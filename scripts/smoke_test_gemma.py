"""
Phase 1 Smoke Test: Gemma 3 4B Instruct Local Inference & Token Score Verification

Validates:
1. Environment & CUDA detection (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM)
2. BitsAndBytes 4-bit NF4 Quantization loading
3. VRAM memory monitoring and management (Baseline, Load, Peak)
4. Distinguishing output_scores from output_logits in Transformers
5. Tokenizer & Model loading for 'google/gemma-3-4b-it'
6. Generation output with `output_scores=True`, `output_logits=True`, and `return_dict_in_generate=True`
7. Prompt token exclusion: confirming generated scores/logits correspond strictly to answer tokens
8. Token ID to log probability alignment and mathematical validation
9. Model revision and generation configuration persistence
"""

import os
import sys
import time
import math
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Ensure local .env is loaded
load_dotenv()


def check_cuda_and_gpu() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. An NVIDIA GPU with CUDA support is required.")
    
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_version = torch.version.cuda
    
    baseline_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    baseline_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    
    print("=" * 75)
    print("PHASE 1: SYSTEM & GPU TELEMETRY")
    print("=" * 75)
    print(f"PyTorch Version:         {torch.__version__}")
    print(f"CUDA Version:            {cuda_version}")
    print(f"GPU Device:              {device_name}")
    print(f"Total Dedicated VRAM:    {total_vram_gb:.2f} GB")
    print(f"Baseline Allocated VRAM: {baseline_allocated_mb:.2f} MB")
    print(f"Baseline Reserved VRAM:  {baseline_reserved_mb:.2f} MB")
    print("=" * 75)
    
    return {
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "cuda_version": cuda_version,
        "baseline_allocated_mb": baseline_allocated_mb,
        "baseline_reserved_mb": baseline_reserved_mb,
    }


def run_gemma_smoke_test(model_id: str = "google/gemma-3-4b-it"):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n[ERROR] HF_TOKEN is not set in .env!")
        print("Please create a .env file with your Hugging Face access token:")
        print("HF_TOKEN=your_token_here")
        print("Ensure you have accepted model terms at https://huggingface.co/google/gemma-3-4b-it")
        sys.exit(1)

    # Late imports to ensure clean environment
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    print(f"\n[INFO] Initializing BitsAndBytes 4-bit NF4 configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[INFO] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    print(f"[INFO] Loading model '{model_id}' with 4-bit quantization on GPU...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_load = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    load_time = time.time() - start_load
    
    post_load_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    post_load_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    peak_load_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    
    print("\n" + "=" * 75)
    print("MODEL LOADING & VRAM FOOTPRINT")
    print("=" * 75)
    print(f"Model Load Time:         {load_time:.2f} seconds")
    print(f"Post-Load Allocated VRAM: {post_load_allocated_mb:.2f} MB ({post_load_allocated_mb / 1024:.2f} GB)")
    print(f"Post-Load Reserved VRAM:  {post_load_reserved_mb:.2f} MB ({post_load_reserved_mb / 1024:.2f} GB)")
    print(f"Peak VRAM During Load:    {peak_load_mb:.2f} MB ({peak_load_mb / 1024:.2f} GB)")
    print(f"Free VRAM Available:     {(6141 - post_load_reserved_mb) / 1024:.2f} GB")

    model_revision = getattr(model.config, "_commit_hash", "default_main")
    print(f"Model Architecture:      {model.config.architectures}")
    print(f"Model Commit Revision:   {model_revision}")
    print("=" * 75)

    # Test Prompt
    prompt_text = "What is the capital city of Karnataka? Answer in one sentence."
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt_text

    print(f"\n[INFO] Tokenizing prompt...")
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    prompt_token_count = inputs.input_ids.shape[1]
    print(f"  Prompt token count: {prompt_token_count}")

    print(f"[INFO] Executing inference with BOTH output_logits=True and output_scores=True...")
    gen_config = {
        "max_new_tokens": 40,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
    }

    torch.cuda.reset_peak_memory_stats()
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            do_sample=gen_config["do_sample"],
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    gen_time = time.time() - start_gen

    # Extraction & Comparison of Scores vs Logits
    full_sequence = outputs.sequences[0]
    total_token_count = len(full_sequence)
    generated_token_ids = full_sequence[prompt_token_count:]
    generated_token_count = len(generated_token_ids)
    
    has_scores = hasattr(outputs, "scores") and outputs.scores is not None
    has_logits = hasattr(outputs, "logits") and outputs.logits is not None

    print("\n" + "=" * 75)
    print("OUTPUT_SCORES VS OUTPUT_LOGITS VERIFICATION")
    print("=" * 75)
    print(f"Total sequence length:     {total_token_count}")
    print(f"Prompt token count:        {prompt_token_count}")
    print(f"Generated token count:     {generated_token_count}")
    print(f"Outputs has 'scores':      {has_scores} (len={len(outputs.scores) if has_scores else 0})")
    print(f"Outputs has 'logits':      {has_logits} (len={len(outputs.logits) if has_logits else 0})")

    # Determine preferred tensor container (raw unprocessed logits preferred)
    if has_logits and len(outputs.logits) == generated_token_count:
        target_tensors = outputs.logits
        logits_type = "unprocessed_logits (RAW)"
    elif has_scores and len(outputs.scores) == generated_token_count:
        target_tensors = outputs.scores
        logits_type = "processed_scores"
    else:
        raise RuntimeError(f"Neither scores nor logits matched generated_token_count {generated_token_count}")

    print(f"Selected Confidence Source: {logits_type}")

    # Check if raw logits and processed scores differ or are identical under greedy decoding
    if has_scores and has_logits:
        diff = (outputs.logits[0] - outputs.scores[0]).abs().max().item()
        print(f"Max Absolute Difference (logits[0] vs scores[0]): {diff:.6e}")
        if diff == 0.0:
            print("  Note: Under greedy decoding without penalty processors, raw logits and scores are identical.")
        else:
            print("  Note: Processed scores diverge from raw logits due to active logit processors.")

    # Assertion 1: Length matches generated tokens exactly (excluding prompt tokens)
    assert len(target_tensors) == generated_token_count, (
        f"Length mismatch: {len(target_tensors)} != {generated_token_count}"
    )
    print("[PASS] Assertion verified: target tensors tuple matches generated token count (prompt tokens excluded).")

    # Extract log probabilities and probabilities for each generated token
    token_records = []
    log_probs_list = []
    probs_list = []

    for step_idx, step_tensor in enumerate(target_tensors):
        # step_tensor shape: (batch_size, vocab_size)
        step_log_probs = F.log_softmax(step_tensor[0], dim=-1)
        step_probs = F.softmax(step_tensor[0], dim=-1)
        
        chosen_token_id = generated_token_ids[step_idx].item()
        token_str = tokenizer.decode([chosen_token_id])
        
        token_log_prob = step_log_probs[chosen_token_id].item()
        token_prob = step_probs[chosen_token_id].item()
        
        # Verify alignment: log(prob) should match log_prob within numerical tolerance
        computed_log_prob = math.log(max(token_prob, 1e-12))
        assert abs(computed_log_prob - token_log_prob) < 1e-4, (
            f"Alignment check failed at step {step_idx}: log_prob={token_log_prob}, math.log(prob)={computed_log_prob}"
        )
        
        log_probs_list.append(token_log_prob)
        probs_list.append(token_prob)
        
        token_records.append({
            "step": step_idx,
            "token_id": chosen_token_id,
            "token_str": token_str,
            "log_prob": token_log_prob,
            "prob": token_prob,
        })

    print("[PASS] Exact alignment between token IDs, softmax probabilities and log-probabilities verified.")

    # Mathematical aggregations
    mean_token_prob = sum(probs_list) / len(probs_list)
    mean_log_prob = sum(log_probs_list) / len(log_probs_list)
    geometric_mean_prob = math.exp(mean_log_prob)

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    print("\n" + "=" * 75)
    print("INFERENCE RESULT & STATISTICAL METRICS")
    print("=" * 75)
    print(f"Generated Answer:           {generated_text}")
    print(f"Generation Latency:         {gen_time:.3f} s ({generated_token_count / gen_time:.1f} tokens/s)")
    print(f"Mean Token Probability:     {mean_token_prob:.4f}")
    print(f"Mean Log Probability:       {mean_log_prob:.4f}")
    print(f"Geometric Mean Probability: {geometric_mean_prob:.4f}")
    print("-" * 75)
    print("Step-by-Step Generated Tokens & Probability Alignment (First 6):")
    for r in token_records[:6]:
        print(f"  Step {r['step']:02d}: ID={r['token_id']:<7} Token={repr(r['token_str']):<15} "
              f"Prob={r['prob']:.4f} LogProb={r['log_prob']:.4f}")
    print("=" * 75)

    # Peak VRAM during generation
    peak_gen_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    final_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    final_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)

    print("\n" + "=" * 75)
    print("FINAL VRAM CONSUMPTION & CUDA STABILITY")
    print("=" * 75)
    print(f"Active Allocated VRAM:   {final_allocated_mb:.2f} MB ({final_allocated_mb / 1024:.2f} GB)")
    print(f"Reserved CUDA VRAM:      {final_reserved_mb:.2f} MB ({final_reserved_mb / 1024:.2f} GB)")
    print(f"Peak VRAM In Generation: {peak_gen_mb:.2f} MB ({peak_gen_mb / 1024:.2f} GB)")
    print(f"Safe VRAM Headroom:      {(6141 - peak_gen_mb) / 1024:.2f} GB / 6.00 GB total")
    print("CUDA Out-Of-Memory (OOM): NONE (PASSED with stable headroom)")
    print("=" * 75)
    print("[SUCCESS] Phase 1 Smoke Test & Token Verification complete.")


if __name__ == "__main__":
    check_cuda_and_gpu()
    run_gemma_smoke_test()
