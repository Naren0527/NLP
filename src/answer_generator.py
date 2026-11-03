"""
Paired Answer Generator for Cross-Lingual Calibration.

Coordinates two independent generation calls against the same underlying Gemma 3 4B model:
1. Call 1: Original-Language Generation
2. Call 2: English Generation

Phase 3 Addition:
3. Elicits verbalized confidence matching each answer's language:
   - Original Answer -> Localized confidence prompt (e.g. v1.0_confidence_kn)
   - English Answer -> English confidence prompt (v1.0_confidence_en)
"""

from dataclasses import dataclass
import os
from typing import Dict, Any, Optional, Tuple

import yaml

from src.model_engine import ModelEngine, GenerationOutput
from src.confidence import ConfidenceEngine, VerbalizedConfidenceResult
from src.language_detector import SUPPORTED_LANGUAGES


@dataclass
class PairedAnswerResult:
    raw_question: str
    detected_language_code: str
    detected_language_name: str
    original_answer: GenerationOutput
    english_answer: GenerationOutput
    original_prompt_version: str
    english_prompt_version: str
    original_confidence: Optional[VerbalizedConfidenceResult] = None
    english_confidence: Optional[VerbalizedConfidenceResult] = None


class AnswerGenerator:
    """
    Orchestrates paired independent generations in the original language and English,
    and elicits independent verbalized confidence for each answer.
    """

    def __init__(
        self,
        model_engine: Optional[ModelEngine] = None,
        prompts_path: str = "configs/prompts.yaml"
    ):
        self.model_engine = model_engine or ModelEngine.get_instance()
        self.prompts_path = prompts_path
        self.prompts_config = self._load_prompts(prompts_path)
        self.confidence_engine = ConfidenceEngine(
            model_engine=self.model_engine,
            prompts_path=prompts_path
        )

    @staticmethod
    def _load_prompts(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def format_original_prompt(self, question: str, language_name: str) -> Tuple[str, str]:
        """Formats the original-language generation prompt with versioning."""
        cfg = self.prompts_config.get("prompt_versions", {}).get("original_language", {})
        version = cfg.get("version", "v1.0_original")
        template = cfg.get(
            "template",
            "You are answering a user question. The user's language is {LANGUAGE}. "
            "Answer only in {LANGUAGE}. Preserve names, numbers, dates and negation. "
            "Do not add unsupported facts. If uncertain, say so. Question: {QUESTION}"
        )
        prompt_text = template.format(LANGUAGE=language_name, QUESTION=question.strip())
        return prompt_text, version

    def format_english_prompt(self, question: str) -> Tuple[str, str]:
        """Formats the English generation prompt with versioning."""
        cfg = self.prompts_config.get("prompt_versions", {}).get("english", {})
        version = cfg.get("version", "v1.0_english")
        template = cfg.get(
            "template",
            "Answer the same question in English, preserving the meaning of the original question. "
            "Be concise and factual. Do not assume the answer in another language is correct. "
            "If uncertain, state uncertainty. Original question: {QUESTION}"
        )
        prompt_text = template.format(QUESTION=question.strip())
        return prompt_text, version

    def generate_paired_answers(
        self,
        question: str,
        language_code: str,
        language_name: Optional[str] = None,
        elicit_confidence: bool = True
    ) -> PairedAnswerResult:
        """
        Executes two independent generation calls:
        Call 1: In the original detected language.
        Call 2: In English.

        Phase 3:
        Elicits independent verbalized confidence matching each answer's language.
        """
        lang_code = language_code.lower().strip()
        lang_name = language_name or SUPPORTED_LANGUAGES.get(lang_code, "English")

        # 1. Format generation prompts
        orig_prompt, orig_version = self.format_original_prompt(question, lang_name)
        eng_prompt, eng_version = self.format_english_prompt(question)

        # 2. Call 1: Original Language Generation
        orig_output = self.model_engine.generate(orig_prompt)

        # 3. Call 2: Independent English Generation
        eng_output = self.model_engine.generate(eng_prompt)

        # 4. Verbalized Confidence Elicitation (if enabled)
        orig_conf = None
        eng_conf = None
        if elicit_confidence:
            # Original answer gets confidence prompt in original language
            orig_conf = self.confidence_engine.elicit_verbalized_confidence(
                question=question,
                answer=orig_output.answer_text,
                answer_language=lang_code
            )
            # English answer gets confidence prompt in English
            eng_conf = self.confidence_engine.elicit_verbalized_confidence(
                question=question,
                answer=eng_output.answer_text,
                answer_language="en"
            )

        return PairedAnswerResult(
            raw_question=question,
            detected_language_code=lang_code,
            detected_language_name=lang_name,
            original_answer=orig_output,
            english_answer=eng_output,
            original_prompt_version=orig_version,
            english_prompt_version=eng_version,
            original_confidence=orig_conf,
            english_confidence=eng_conf,
        )
