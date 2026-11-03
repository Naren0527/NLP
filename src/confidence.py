"""
Confidence Engine for Cross-Lingual Calibration.

Implements decoupled confidence measurement:
1. Verbalized Confidence: Elicited via separate localized calls using a strict
   percentage output contract (r"^\\s*(\\d+(?:\\.\\d+)?)\\s*%\\s*$") and normalized to [0.0, 1.0].
2. Statistical Confidence: Derived from float32 log-softmax of raw generation logits.
   Maintains two independent metric sets:
   - Content-Token Metrics: Excludes verified special/control tokens derived from tokenizer metadata.
   - Full-Sequence Metrics: Evaluates all generated tokens.
3. Strict Mathematical Consistency: Asserts geom_prob == exp(mean_log_prob) and
   labels metrics as token-generation statistics, never factual correctness probabilities.
"""

from dataclasses import dataclass, field
import math
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
import torch.nn.functional as F
import yaml

from src.language_detector import SUPPORTED_LANGUAGES


def normalize_unicode_digits(text: str) -> str:
    """
    Normalizes Indic and other Unicode decimal digits to ASCII digits '0'-'9'.
    Preserves all other characters intact.
    """
    if not text:
        return ""
    result = []
    for ch in text:
        if ch.isdecimal():
            # unicodedata.digit converts any Unicode decimal character (e.g. ೮, ८, ௮) to int 0-9
            result.append(str(unicodedata.digit(ch)))
        else:
            result.append(ch)
    return "".join(result)


@dataclass
class TokenMetric:
    step: int
    token_id: int
    token_str: str
    prob: float
    log_prob: float
    is_special: bool


@dataclass
class MetricSummary:
    mean_token_probability: float          # Arithmetic mean of probs: (1/K) * sum(p_i)
    mean_log_probability: float            # Arithmetic mean of log-probs: (1/K) * sum(ln p_i)
    geometric_probability: float           # exp(mean_log_prob)
    tokens_evaluated: int                  # Count of tokens in this subset (K)
    token_indices: List[int]               # Exact step indices included in calculation


@dataclass
class StatisticalConfidenceResult:
    content_tokens: MetricSummary          # Special tokens excluded
    full_sequence: MetricSummary           # All generated tokens included
    total_tokens_generated: int
    special_token_count: int
    special_token_ids_found: List[int]
    special_token_policy: str              # E.g., 'tokenizer_all_special_ids'
    token_metrics: List[TokenMetric]
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class VerbalizedConfidenceResult:
    raw_response: str
    confidence_score: Optional[float]      # Normalized strictly to [0.0, 1.0] (e.g. 0.85)
    confidence_percent: Optional[float]    # In [0.0, 100.0] (e.g. 85.0)
    is_valid: bool
    parsing_error: Optional[str]           # Explicit error code if invalid
    prompt_version: str
    elicitation_prompt: str
    prompt_language: str
    answer_language: str
    confidence_call_config: Dict[str, Any]


@dataclass
class ConfidenceReport:
    statistical: StatisticalConfidenceResult
    verbalized: VerbalizedConfidenceResult
    logits_type: str                       # 'unprocessed_logits (RAW)'
    extraction_status: str                 # 'SUCCESS', 'VERBALIZED_FAILED_STATISTICAL_OK', etc.


class VerbalizedConfidenceParser:
    """
    Strict percentage parser enforcing full-string match: r"^\\s*(\\d+(?:\\.\\d+)?)\\s*%\\s*$"
    """
    STRICT_PERCENT_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")

    @classmethod
    def parse(
        cls,
        raw_response: str,
        prompt_version: str = "unknown",
        elicitation_prompt: str = "",
        prompt_language: str = "en",
        answer_language: str = "en",
        confidence_call_config: Optional[Dict[str, Any]] = None
    ) -> VerbalizedConfidenceResult:
        config = confidence_call_config or {}
        raw_str = raw_response if raw_response is not None else ""

        if not raw_str.strip():
            return VerbalizedConfidenceResult(
                raw_response=raw_str,
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error="EMPTY_RESPONSE",
                prompt_version=prompt_version,
                elicitation_prompt=elicitation_prompt,
                prompt_language=prompt_language,
                answer_language=answer_language,
                confidence_call_config=config,
            )

        # Normalize Unicode decimal digits to ASCII '0'-'9'
        normalized_str = normalize_unicode_digits(raw_str).strip()

        # Full-string regex match test
        match = cls.STRICT_PERCENT_REGEX.match(normalized_str)
        if not match:
            # Diagnose specific failure reason for research provenance
            error_code = cls._diagnose_failure(normalized_str)
            return VerbalizedConfidenceResult(
                raw_response=raw_str,
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error=error_code,
                prompt_version=prompt_version,
                elicitation_prompt=elicitation_prompt,
                prompt_language=prompt_language,
                answer_language=answer_language,
                confidence_call_config=config,
            )

        num_str = match.group(1)
        try:
            val = float(num_str)
        except ValueError:
            return VerbalizedConfidenceResult(
                raw_response=raw_str,
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error="NUMERIC_PARSE_ERROR",
                prompt_version=prompt_version,
                elicitation_prompt=elicitation_prompt,
                prompt_language=prompt_language,
                answer_language=answer_language,
                confidence_call_config=config,
            )

        if val < 0.0 or val > 100.0:
            return VerbalizedConfidenceResult(
                raw_response=raw_str,
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error=f"OUT_OF_RANGE_{val}",
                prompt_version=prompt_version,
                elicitation_prompt=elicitation_prompt,
                prompt_language=prompt_language,
                answer_language=answer_language,
                confidence_call_config=config,
            )

        # Success: normalized to [0.0, 1.0]
        norm_score = round(val / 100.0, 6)
        return VerbalizedConfidenceResult(
            raw_response=raw_str,
            confidence_score=norm_score,
            confidence_percent=val,
            is_valid=True,
            parsing_error=None,
            prompt_version=prompt_version,
            elicitation_prompt=elicitation_prompt,
            prompt_language=prompt_language,
            answer_language=answer_language,
            confidence_call_config=config,
        )

    @classmethod
    def _diagnose_failure(cls, normalized_str: str) -> str:
        """Determines the exact failure cause when strict regex does not match."""
        # 1. Negative percentage (e.g. "-5%")
        if re.match(r"^\s*-\d+(?:\.\d+)?\s*%\s*$", normalized_str):
            return "OUT_OF_RANGE_NEGATIVE"

        # 2. Bare number (e.g. "85", "1", "1.0", "0.85")
        if re.match(r"^\s*-?\d+(?:\.\d+)?\s*$", normalized_str):
            return "BARE_NUMBER_MISSING_PERCENT_SYMBOL"
        
        # 3. Multiple percentages
        if len(re.findall(r"%", normalized_str)) > 1:
            return "MULTIPLE_PERCENTAGES_FOUND"
        
        # 4. Contains percent but surrounded by prose/words
        if "%" in normalized_str:
            return "PROSE_OR_EXPLANATION_PRESENT"

        # 5. Completely non-numeric or malformed
        return "MALFORMED_NON_PERCENTAGE_RESPONSE"


class StatisticalConfidenceCalculator:
    """
    Computes rigorous statistical sequence metrics from raw generation logits.
    Enforces float32 log-softmax precision and isolates content vs full sequence tokens.
    """

    @classmethod
    def compute(
        cls,
        step_logits: List[torch.Tensor],
        generated_token_ids: List[int],
        special_token_ids: Set[int],
        tokenizer: Optional[Any] = None
    ) -> StatisticalConfidenceResult:
        total_tokens = len(generated_token_ids)
        if total_tokens == 0 or len(step_logits) == 0:
            empty_summary = MetricSummary(0.0, 0.0, 0.0, 0, [])
            return StatisticalConfidenceResult(
                content_tokens=empty_summary,
                full_sequence=empty_summary,
                total_tokens_generated=0,
                special_token_count=0,
                special_token_ids_found=[],
                special_token_policy="empty_sequence",
                token_metrics=[],
                is_valid=False,
                error_message="EMPTY_TOKEN_SEQUENCE"
            )

        if len(step_logits) != total_tokens:
            raise ValueError(
                f"Mismatch between step logits ({len(step_logits)}) and generated tokens ({total_tokens})."
            )

        token_metrics: List[TokenMetric] = []
        special_found: List[int] = []

        for step_idx, logits_tensor in enumerate(step_logits):
            token_id = generated_token_ids[step_idx]

            # Enforce float32 precision for log-softmax calculation
            # logits_tensor can be (batch_size, vocab) or (vocab,)
            step_tensor = logits_tensor[0] if logits_tensor.dim() > 1 else logits_tensor
            step_tensor_f32 = step_tensor.to(torch.float32)

            # Check finite logits
            if not torch.isfinite(step_tensor_f32).all():
                raise ValueError(f"Non-finite logits detected at generation step {step_idx}.")

            vocab_size = step_tensor_f32.shape[-1]
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"Token ID {token_id} at step {step_idx} is out of vocabulary bounds [0, {vocab_size})."
                )

            # Primary numerical representation: selected-token log-probability
            step_log_probs = F.log_softmax(step_tensor_f32, dim=-1)
            t_log_prob = step_log_probs[token_id].item()

            # Derived probability via exponentiation
            t_prob = math.exp(t_log_prob)

            is_special = token_id in special_token_ids
            if is_special:
                special_found.append(token_id)

            t_str = tokenizer.decode([token_id]) if tokenizer is not None else f"id_{token_id}"

            token_metrics.append(TokenMetric(
                step=step_idx,
                token_id=token_id,
                token_str=t_str,
                prob=t_prob,
                log_prob=t_log_prob,
                is_special=is_special,
            ))

        # 1. Full Sequence Metrics
        full_indices = list(range(total_tokens))
        full_summary = cls._summarize_subset(token_metrics, full_indices)

        # 2. Content Token Metrics (Special tokens excluded)
        content_indices = [m.step for m in token_metrics if not m.is_special]
        content_summary = cls._summarize_subset(token_metrics, content_indices)

        return StatisticalConfidenceResult(
            content_tokens=content_summary,
            full_sequence=full_summary,
            total_tokens_generated=total_tokens,
            special_token_count=len(special_found),
            special_token_ids_found=special_found,
            special_token_policy="tokenizer_all_special_ids",
            token_metrics=token_metrics,
            is_valid=True,
            error_message=None
        )

    @classmethod
    def _summarize_subset(cls, metrics: List[TokenMetric], indices: List[int]) -> MetricSummary:
        if not indices:
            return MetricSummary(
                mean_token_probability=0.0,
                mean_log_probability=0.0,
                geometric_probability=0.0,
                tokens_evaluated=0,
                token_indices=[]
            )

        selected = [metrics[i] for i in indices]
        k = len(selected)
        
        # Arithmetic mean of probabilities
        mean_prob = sum(m.prob for m in selected) / k
        
        # Arithmetic mean of log-probabilities
        mean_log_prob = sum(m.log_prob for m in selected) / k
        
        # Geometric mean probability via exp(mean_log_prob)
        geom_prob = math.exp(mean_log_prob)

        # Mathematical verification: ln(geom_prob) == mean_log_prob
        # Within numerical tolerance
        assert abs(math.log(max(geom_prob, 1e-12)) - mean_log_prob) < 1e-5, (
            f"Mathematical inconsistency: ln({geom_prob}) != {mean_log_prob}"
        )

        return MetricSummary(
            mean_token_probability=round(mean_prob, 6),
            mean_log_probability=round(mean_log_prob, 6),
            geometric_probability=round(geom_prob, 6),
            tokens_evaluated=k,
            token_indices=indices
        )


class ConfidenceEngine:
    """
    High-level orchestrator managing:
    - Independent verbalized confidence elicitation matching the answer language.
    - Dual statistical confidence calculation (content vs full).
    - Assembling the comprehensive ConfidenceReport.
    """

    def __init__(self, model_engine: Optional[Any] = None, prompts_path: str = "configs/prompts.yaml"):
        self.model_engine = model_engine
        self.prompts_path = prompts_path
        self.prompts_config = self._load_prompts(prompts_path)

    @staticmethod
    def _load_prompts(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_confidence_prompt(
        self,
        question: str,
        answer: str,
        answer_language: str
    ) -> Tuple[str, str, str]:
        """
        Formats the confidence prompt matching the exact language of the answer.
        Returns: (prompt_text, prompt_version, prompt_language)
        """
        lang = answer_language.lower().strip()
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"

        prompt_key = f"confidence_{lang}"
        cfg = self.prompts_config.get("prompt_versions", {}).get(prompt_key, {})
        version = cfg.get("version", f"v1.0_confidence_{lang}")
        template = cfg.get("template")

        if not template:
            # Fallback to English template if specific language template missing
            cfg_en = self.prompts_config.get("prompt_versions", {}).get("confidence_en", {})
            version = cfg_en.get("version", "v1.0_confidence_en")
            template = cfg_en.get(
                "template",
                "You are evaluating the factual correctness of an answer.\n"
                "Question: {QUESTION}\nAnswer: {ANSWER}\n"
                "Respond with exactly one percentage between 0% and 100% in the format NN% (for example: 85%).\n"
                "Confidence:"
            )

        prompt_text = template.format(QUESTION=question.strip(), ANSWER=answer.strip())
        return prompt_text, version, lang

    def elicit_verbalized_confidence(
        self,
        question: str,
        answer: str,
        answer_language: str
    ) -> VerbalizedConfidenceResult:
        """
        Executes a separate forward pass against Gemma 3 4B using greedy decoding
        to elicit verbalized confidence under the strict NN% contract.
        """
        prompt_text, prompt_version, prompt_lang = self.get_confidence_prompt(
            question=question,
            answer=answer,
            answer_language=answer_language
        )

        call_config = {
            "max_new_tokens": 10,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }

        if self.model_engine is None:
            # When model engine is not loaded, return structured uninitialized result
            return VerbalizedConfidenceResult(
                raw_response="",
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error="MODEL_ENGINE_NOT_INITIALIZED",
                prompt_version=prompt_version,
                elicitation_prompt=prompt_text,
                prompt_language=prompt_lang,
                answer_language=answer_language,
                confidence_call_config=call_config,
            )

        try:
            # Call generation with max_new_tokens=10
            gen_out = self.model_engine.generate(
                prompt=prompt_text,
                max_new_tokens=call_config["max_new_tokens"],
                apply_template=True
            )
            raw_response = gen_out.answer_text
        except Exception as e:
            return VerbalizedConfidenceResult(
                raw_response="",
                confidence_score=None,
                confidence_percent=None,
                is_valid=False,
                parsing_error=f"GENERATION_EXCEPTION: {str(e)}",
                prompt_version=prompt_version,
                elicitation_prompt=prompt_text,
                prompt_language=prompt_lang,
                answer_language=answer_language,
                confidence_call_config=call_config,
            )

        # Parse output strictly
        return VerbalizedConfidenceParser.parse(
            raw_response=raw_response,
            prompt_version=prompt_version,
            elicitation_prompt=prompt_text,
            prompt_language=prompt_lang,
            answer_language=answer_language,
            confidence_call_config=call_config,
        )

    def extract_confidence_report(
        self,
        question: str,
        answer_text: str,
        answer_language: str,
        step_logits: List[torch.Tensor],
        generated_token_ids: List[int],
        special_token_ids: Set[int],
        logits_type: str = "unprocessed_logits (RAW)",
        tokenizer: Optional[Any] = None
    ) -> ConfidenceReport:
        """
        Builds the unified confidence report combining:
        1. Dual statistical metrics (content tokens vs full sequence).
        2. Independent verbalized confidence elicitation matching the answer language.
        """
        # 1. Statistical Confidence
        statistical_res = StatisticalConfidenceCalculator.compute(
            step_logits=step_logits,
            generated_token_ids=generated_token_ids,
            special_token_ids=special_token_ids,
            tokenizer=tokenizer
        )

        # 2. Verbalized Confidence
        verbalized_res = self.elicit_verbalized_confidence(
            question=question,
            answer=answer_text,
            answer_language=answer_language
        )

        # Extraction status
        if statistical_res.is_valid and verbalized_res.is_valid:
            status = "SUCCESS"
        elif statistical_res.is_valid and not verbalized_res.is_valid:
            status = "VERBALIZED_FAILED_STATISTICAL_OK"
        elif not statistical_res.is_valid and verbalized_res.is_valid:
            status = "STATISTICAL_FAILED_VERBALIZED_OK"
        else:
            status = "BOTH_FAILED"

        return ConfidenceReport(
            statistical=statistical_res,
            verbalized=verbalized_res,
            logits_type=logits_type,
            extraction_status=status
        )
