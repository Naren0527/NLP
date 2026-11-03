"""
Storage Engine for Cross-Lingual Calibration.

Handles immutable, append-only JSONL recording of experimental generations adhering
strictly to the blueprint schema (Section 10) and reproducibility requirements.
Preserves all Phase 2 fields and incorporates Phase 3 dual statistical and verbalized
confidence fields non-destructively.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import sys
import uuid
from typing import Dict, Any, List, Optional, Tuple

import torch
import transformers

from src.answer_generator import PairedAnswerResult
from src.language_detector import DetectionResult
from src.model_engine import GenerationOutput
from src.confidence import VerbalizedConfidenceResult


class StorageEngine:
    """
    Manages JSONL persistence for experimental generations.
    """

    def __init__(self, output_path: str = "results/predictions.jsonl"):
        self.output_path = output_path
        self._ensure_directory(output_path)

    @staticmethod
    def _ensure_directory(path: str) -> None:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def get_software_versions() -> Dict[str, str]:
        """Returns frozen environment software versions for reproducibility."""
        import bitsandbytes as bnb
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "bitsandbytes": getattr(bnb, "__version__", "unknown"),
        }

    def create_record(
        self,
        run_id: str,
        original_question: str,
        detection_result: DetectionResult,
        answer_language: str,
        answer_stage: str,
        prompt_version: str,
        gen_output: GenerationOutput,
        verbalized_conf: Optional[VerbalizedConfidenceResult] = None,
        consistency_label: Optional[str] = None,
        comparison_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Constructs a fully populated dictionary matching the audit schema."""
        # 1. Determine confidence extraction status
        stat_valid = (
            gen_output.statistical_confidence is not None
            and gen_output.statistical_confidence.is_valid
        )
        verb_valid = verbalized_conf is not None and verbalized_conf.is_valid

        if stat_valid and verb_valid:
            extraction_status = "SUCCESS"
        elif stat_valid and not verb_valid:
            extraction_status = "VERBALIZED_FAILED_STATISTICAL_OK"
        elif not stat_valid and verb_valid:
            extraction_status = "STATISTICAL_FAILED_VERBALIZED_OK"
        else:
            extraction_status = "CONFIDENCE_UNEVALUATED"

        # 2. Format statistical confidence breakdown
        stat_data = None
        if gen_output.statistical_confidence:
            sc = gen_output.statistical_confidence
            stat_data = {
                "content_tokens": asdict(sc.content_tokens),
                "full_sequence": asdict(sc.full_sequence),
                "special_tokens_excluded": sc.special_token_ids_found,
                "special_token_count": sc.special_token_count,
                "special_token_policy": sc.special_token_policy,
            }

        # 3. Construct dictionary preserving ALL Phase 2 & Phase 3 fields
        return {
            # Core Identifiers
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": gen_output.model_id,
            "model_revision": gen_output.model_revision,
            "quantization": "4bit-nf4-double_quant",
            "device_info": gen_output.device_info,
            "software_versions": self.get_software_versions(),
            
            # Input & Language Metadata
            "original_question": original_question,
            "detected_language": detection_result.language_code,
            "detected_language_name": detection_result.language_name,
            "language_detection_confidence": detection_result.confidence,
            "is_language_overridden": detection_result.is_overridden,
            "language_detection_method": detection_result.detection_method,
            "answer_language": answer_language,
            "answer_stage": answer_stage,

            # Generation Details & Prompts
            "prompt_version": prompt_version,
            "exact_prompt": gen_output.prompt_text,
            "formatted_prompt": gen_output.formatted_prompt,
            "answer_text": gen_output.answer_text,
            
            # Preserved Phase 2 Statistical Fields
            "token_count": gen_output.generated_token_count,
            "prompt_token_count": gen_output.prompt_token_count,
            "total_token_count": gen_output.total_token_count,
            "generated_token_ids": gen_output.generated_token_ids,
            "token_logprobs": gen_output.token_logprobs,
            "mean_token_probability": gen_output.mean_token_probability,
            "mean_log_probability": gen_output.mean_log_probability,
            "geometric_probability": gen_output.geometric_probability,
            "logits_type": gen_output.logits_type,
            "execution_time_seconds": gen_output.execution_time_seconds,
            "generation_configuration": gen_output.generation_config,

            # Phase 3: Dual Statistical Metrics (Content vs Full)
            "statistical_confidence": stat_data,

            # Phase 3: Populated Verbalized Confidence Fields
            "verbalized_confidence_raw": verbalized_conf.raw_response if verbalized_conf else None,
            "verbalized_confidence": verbalized_conf.confidence_score if verbalized_conf else None,
            "verbalized_confidence_percent": verbalized_conf.confidence_percent if verbalized_conf else None,
            "verbalized_confidence_valid": verbalized_conf.is_valid if verbalized_conf else False,
            "verbalized_confidence_error": verbalized_conf.parsing_error if verbalized_conf else None,
            "verbalized_prompt_version": verbalized_conf.prompt_version if verbalized_conf else None,
            "verbalized_prompt_language": verbalized_conf.prompt_language if verbalized_conf else None,
            "verbalized_call_config": verbalized_conf.confidence_call_config if verbalized_conf else None,
            "confidence_extraction_status": extraction_status,

            # Phase 4: Answer Comparison & Consistency Fields
            "consistency_label": consistency_label,
            "comparison_details": comparison_details,

            # Placeholders for future phases (Phase 5+)
            "correctness_label": None,
            "reference_answer": None,
            "evidence": None,
            "verification_status": "unverified",
            "mitigation_trigger": None,
            "mitigation_status": None,
        }

    def append_record(self, record: Dict[str, Any]) -> None:
        """Appends a single JSON record to the JSONL file."""
        self._ensure_directory(self.output_path)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_paired_result(
        self,
        paired_result: PairedAnswerResult,
        detection_result: DetectionResult,
        run_id: Optional[str] = None,
        answer_stage: str = "initial",
        comparison_result: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Saves both original-language and English answers as separate records.
        Attaches the corresponding localized verbalized confidence result to each record.
        If comparison_result is provided, attaches consistency_label and comparison_details.
        """
        actual_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

        consistency_label = None
        comparison_details = None
        if comparison_result is not None:
            consistency_label = getattr(comparison_result, "final_arbiter_classification", None)
            if hasattr(comparison_result, "comparison_id"):
                comparison_details = {
                    "comparison_id": comparison_result.comparison_id,
                    "raw_judge_classification": comparison_result.raw_judge_classification,
                    "final_arbiter_classification": comparison_result.final_arbiter_classification,
                    "classification_decision_source": comparison_result.classification_decision_source,
                    "arbiter_override_reason": comparison_result.arbiter_override_reason,
                    "overall_rationale": comparison_result.overall_rationale,
                    "evidence_validation_status": comparison_result.evidence_validation_status,
                    "claims": [asdict(c) for c in comparison_result.claims],
                    "deterministic_features": asdict(comparison_result.deterministic_features),
                    "comparison_prompt_version": comparison_result.prompt_version,
                    "execution_status": comparison_result.execution_status,
                    "execution_time_seconds": comparison_result.execution_time_seconds,
                }

        # Record 1: Original Language
        orig_record = self.create_record(
            run_id=actual_run_id,
            original_question=paired_result.raw_question,
            detection_result=detection_result,
            answer_language=paired_result.detected_language_code,
            answer_stage=answer_stage,
            prompt_version=paired_result.original_prompt_version,
            gen_output=paired_result.original_answer,
            verbalized_conf=paired_result.original_confidence,
            consistency_label=consistency_label,
            comparison_details=comparison_details,
        )
        self.append_record(orig_record)

        # Record 2: English
        eng_record = self.create_record(
            run_id=actual_run_id,
            original_question=paired_result.raw_question,
            detection_result=detection_result,
            answer_language="en",
            answer_stage=answer_stage,
            prompt_version=paired_result.english_prompt_version,
            gen_output=paired_result.english_answer,
            verbalized_conf=paired_result.english_confidence,
            consistency_label=consistency_label,
            comparison_details=comparison_details,
        )
        self.append_record(eng_record)

        return orig_record, eng_record
