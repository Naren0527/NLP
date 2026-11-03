"""
Model Engine for Gemma 3 4B Instruct.

Reuses the exact Phase 1 verified architecture and configuration:
- Model: google/gemma-3-4b-it (Architecture: Gemma3ForConditionalGeneration)
- Quantization: BitsAndBytes 4-bit NF4 with double quantization and bfloat16 compute
- Device Map: auto (CUDA:0)
- Generation: Greedy decoding (do_sample=False)
- Extraction: Raw generation logits (output_logits=True) with prompt token exclusion
- Log-softmax computed in Float32 precision for numerical stability
- Singleton management: Model loaded once into VRAM (~3.03 GB) and reused across calls
- Resolved Model Revision: Captures the actual 40-character Git commit hash
"""

import os
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

from src.confidence import (
    StatisticalConfidenceCalculator,
    StatisticalConfidenceResult,
    TokenMetric,
    MetricSummary,
)

# Load local environment
load_dotenv()


@dataclass
class TokenScoreRecord:
    step: int
    token_id: int
    token_str: str
    log_prob: float
    prob: float


@dataclass
class GenerationOutput:
    prompt_text: str
    formatted_prompt: str
    answer_text: str
    prompt_token_count: int
    generated_token_count: int
    total_token_count: int
    generated_token_ids: List[int]
    token_scores: List[TokenScoreRecord]
    token_logprobs: List[float]
    mean_token_probability: float          # Preserved Phase 2 field (full sequence)
    mean_log_probability: float            # Preserved Phase 2 field (full sequence)
    geometric_probability: float           # Preserved Phase 2 field (full sequence)
    logits_type: str
    execution_time_seconds: float
    model_id: str
    model_revision: str
    device_info: Dict[str, Any]
    generation_config: Dict[str, Any]
    statistical_confidence: Optional[StatisticalConfidenceResult] = None  # Phase 3 dual breakdown


class ModelEngine:
    """
    Singleton-capable model engine managing local GPU inference for Gemma 3 4B Instruct.
    """
    _instance: Optional["ModelEngine"] = None

    def __init__(self, config_path: str = "configs/model.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = self.config.get("model", {}).get("model_id", "google/gemma-3-4b-it")
        self.model_revision = self._resolve_model_revision()
        self.special_token_ids: Set[int] = set()
        self._is_loaded = False

    @classmethod
    def get_instance(cls, config_path: str = "configs/model.yaml") -> "ModelEngine":
        """Returns the active singleton ModelEngine instance or initializes it."""
        if cls._instance is None:
            cls._instance = cls(config_path=config_path)
        return cls._instance

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_model_revision(self) -> str:
        """
        Resolves the actual 40-character Git commit hash from Hugging Face cache metadata.
        Avoids generic placeholders like 'default_main'.
        """
        # 1. Check if model is already loaded and has commit hash in config
        if self.model and hasattr(self.model, "config"):
            commit = getattr(self.model.config, "_commit_hash", None)
            if commit and len(commit) == 40:
                return commit

        # 2. Check cached refs/main file in ~/.cache/huggingface/hub/
        repo_slug = self.model_id.replace("/", "--")
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_slug}"
        ref_main = hub_dir / "refs" / "main"
        if ref_main.exists():
            try:
                commit = ref_main.read_text(encoding="utf-8").strip()
                if len(commit) == 40:
                    return commit
            except Exception:
                pass

        # 3. Check snapshots directory
        snapshots_dir = hub_dir / "snapshots"
        if snapshots_dir.exists():
            try:
                for entry in snapshots_dir.iterdir():
                    if entry.is_dir() and len(entry.name) == 40:
                        return entry.name
            except Exception:
                pass

        return "unknown_revision"

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def load_model(self) -> None:
        """
        Loads Gemma 3 4B Instruct once using 4-bit NF4 quantization.
        Reuses the exact loading mechanism verified in Phase 1 smoke test.
        """
        if self.is_loaded():
            return

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Please configure HF_TOKEN in your local .env file."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. NVIDIA GPU is required for model inference.")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        quant_cfg = self.config.get("model", {}).get("quantization", {})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        
        # Exact verified loading call from smoke test
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.config.get("model", {}).get("device_map", "auto"),
            token=hf_token,
        )

        # Cache special token IDs directly from tokenizer metadata
        if hasattr(self.tokenizer, "all_special_ids") and self.tokenizer.all_special_ids:
            self.special_token_ids = set(self.tokenizer.all_special_ids)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            self.special_token_ids.add(self.tokenizer.eos_token_id)

        self.model_revision = self._resolve_model_revision()
        self._is_loaded = True

    def get_gpu_memory_status(self) -> Dict[str, float]:
        """Returns current GPU memory status in Megabytes."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
            "peak_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        apply_template: bool = True
    ) -> GenerationOutput:
        """
        Executes a single generation call, extracting raw logits and computing token statistics.
        Calculates log-softmax in float32 for numerical stability.
        Computes both content-token and full-sequence statistical metrics.
        """
        if not self.is_loaded():
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        gen_cfg = self.config.get("generation", {})
        actual_max_new_tokens = max_new_tokens or gen_cfg.get("max_new_tokens", 128)

        # Apply official chat template
        if apply_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_token_count = inputs.input_ids.shape[1]

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=actual_max_new_tokens,
                    do_sample=gen_cfg.get("do_sample", False),
                    return_dict_in_generate=True,
                    output_scores=gen_cfg.get("output_scores", True),
                    output_logits=gen_cfg.get("output_logits", True),
                )
        except torch.cuda.OutOfMemoryError as oom_err:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA Out of Memory during generation with max_new_tokens={actual_max_new_tokens}. "
                f"Peak allocated: {torch.cuda.max_memory_allocated(0)/(1024**2):.1f} MB"
            ) from oom_err

        elapsed_time = time.time() - start_time

        # Extract sequence and slice out prompt tokens
        full_sequence = outputs.sequences[0]
        total_token_count = len(full_sequence)
        generated_token_ids = full_sequence[prompt_token_count:].tolist()
        generated_token_count = len(generated_token_ids)

        # Prefer raw unprocessed logits if returned
        has_logits = hasattr(outputs, "logits") and outputs.logits is not None and len(outputs.logits) > 0
        has_scores = hasattr(outputs, "scores") and outputs.scores is not None and len(outputs.scores) > 0

        if has_logits and len(outputs.logits) == generated_token_count:
            step_tensors = outputs.logits
            logits_type = "unprocessed_logits (RAW)"
        elif has_scores and len(outputs.scores) == generated_token_count:
            step_tensors = outputs.scores
            logits_type = "processed_scores"
        else:
            raise RuntimeError(
                f"Neither logits ({len(outputs.logits) if has_logits else 0}) nor scores "
                f"({len(outputs.scores) if has_scores else 0}) matched generated token count {generated_token_count}."
            )

        # Phase 3: Compute rigorous dual statistical confidence (float32 log-softmax)
        stat_conf = StatisticalConfidenceCalculator.compute(
            step_logits=step_tensors,
            generated_token_ids=generated_token_ids,
            special_token_ids=self.special_token_ids,
            tokenizer=self.tokenizer
        )

        # Build backward-compatible TokenScoreRecord list and logprobs
        token_scores: List[TokenScoreRecord] = []
        token_logprobs: List[float] = []

        for m in stat_conf.token_metrics:
            token_logprobs.append(m.log_prob)
            token_scores.append(TokenScoreRecord(
                step=m.step,
                token_id=m.token_id,
                token_str=m.token_str,
                log_prob=m.log_prob,
                prob=m.prob,
            ))

        # Memory cleanup: remove reference to heavy output tensors immediately
        del outputs
        del step_tensors
        del inputs
        torch.cuda.empty_cache()

        # Full sequence metrics for backwards compatibility
        mean_token_prob = stat_conf.full_sequence.mean_token_probability
        mean_log_prob = stat_conf.full_sequence.mean_log_probability
        geometric_prob = stat_conf.full_sequence.geometric_probability

        # Answer text: strip special tokens (EOS, <end_of_turn>) for clean presentation
        answer_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        device_info = {
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            **self.get_gpu_memory_status()
        }

        return GenerationOutput(
            prompt_text=prompt,
            formatted_prompt=formatted_prompt,
            answer_text=answer_text,
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            total_token_count=total_token_count,
            generated_token_ids=generated_token_ids,
            token_scores=token_scores,
            token_logprobs=token_logprobs,
            mean_token_probability=mean_token_prob,
            mean_log_probability=mean_log_prob,
            geometric_probability=geometric_prob,
            logits_type=logits_type,
            execution_time_seconds=round(elapsed_time, 4),
            model_id=self.model_id,
            model_revision=self.model_revision,
            device_info=device_info,
            generation_config={
                "max_new_tokens": actual_max_new_tokens,
                "do_sample": gen_cfg.get("do_sample", False),
            },
            statistical_confidence=stat_conf
        )
