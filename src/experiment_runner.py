"""
Pipeline Orchestrator and Command-Line Prototype for Cross-Lingual Calibration.

Entry point:
python -m src.experiment_runner

Coordinates:
Input -> Language Detection -> Optional Manual Override -> Paired Answer Generation
-> Phase 3 Dual Confidence Extraction (Statistical + Verbalized) -> JSONL Storage -> Display

Important:
- Model loads once on startup and is preserved in GPU VRAM across multiple queries.
- Raw logits are extracted and recorded; token probabilities are explicitly labeled
  as token-level modeling proxies, NOT factual correctness probabilities.
- Does not implement confidence calibration, comparison, or mitigation in Phase 3.
"""

import io
import sys
import uuid
from typing import Optional, Dict, Any

from src.language_detector import LanguageDetector, DetectionResult, SUPPORTED_LANGUAGES
from src.model_engine import ModelEngine
from src.answer_generator import AnswerGenerator, PairedAnswerResult
from src.confidence import VerbalizedConfidenceResult
from src.answer_comparator import AnswerComparator, ComparisonResult
from src.storage import StorageEngine


class ExperimentRunner:
    """
    Coordinates the end-to-end Phase 4 inference, confidence, and answer comparison pipeline.
    """

    def __init__(
        self,
        model_engine: Optional[ModelEngine] = None,
        output_path: str = "results/predictions.jsonl",
        prompts_path: str = "configs/prompts.yaml"
    ):
        self.output_path = output_path
        self.detector = LanguageDetector()
        self.model_engine = model_engine or ModelEngine.get_instance()
        self.answer_generator = AnswerGenerator(
            model_engine=self.model_engine,
            prompts_path=prompts_path
        )
        self.comparator = AnswerComparator(
            model_engine=self.model_engine,
            prompts_path=prompts_path
        )
        self.storage = StorageEngine(output_path=output_path)

    def run_pipeline(
        self,
        question: str,
        override_lang: Optional[str] = None,
        run_id: Optional[str] = None,
        elicit_confidence: bool = True,
        compare_answers: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the complete pipeline for a single question.
        Generates paired answers, elicits corresponding localized verbalized confidence,
        and executes evidence-grounded factual comparison.
        """
        actual_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

        # 1. Language Detection
        detection = self.detector.detect(question, override_lang=override_lang)
        if detection.is_uncertain:
            return {
                "run_id": actual_run_id,
                "status": "UNCERTAIN_LANGUAGE",
                "detection": detection,
                "paired_result": None,
                "comparison_result": None,
                "records": None,
            }

        assert detection.language_code is not None

        # 2. Paired Answer Generation + Localized Verbalized Confidence
        paired_result = self.answer_generator.generate_paired_answers(
            question=question,
            language_code=detection.language_code,
            language_name=detection.language_name,
            elicit_confidence=elicit_confidence
        )

        # 3. Evidence-Grounded Answer Comparison (Phase 4)
        comparison_res = None
        if compare_answers:
            comparison_res = self.comparator.compare(
                question=question,
                original_answer_text=paired_result.original_answer.answer_text,
                english_answer_text=paired_result.english_answer.answer_text,
                original_language_code=detection.language_code,
                original_language_name=detection.language_name,
                run_id=actual_run_id,
            )

        # 4. Save Experimental Records to JSONL (with comparison data attached)
        orig_rec, eng_rec = self.storage.save_paired_result(
            paired_result=paired_result,
            detection_result=detection,
            run_id=actual_run_id,
            answer_stage="initial",
            comparison_result=comparison_res,
        )

        return {
            "run_id": actual_run_id,
            "status": "SUCCESS",
            "detection": detection,
            "paired_result": paired_result,
            "comparison_result": comparison_res,
            "records": (orig_rec, eng_rec),
        }


def format_confidence_display(gen_output, conf_result) -> str:
    lines = []
    # Statistical breakdown
    if gen_output.statistical_confidence:
        sc = gen_output.statistical_confidence
        c = sc.content_tokens
        f = sc.full_sequence
        lines.append(f"  [Statistical] Content Tokens ({c.tokens_evaluated}/{sc.total_tokens_generated}): "
                     f"Mean Prob={c.mean_token_probability:.4f} | Mean LogProb={c.mean_log_probability:.4f} | "
                     f"Geom Prob={c.geometric_probability:.4f}")
        lines.append(f"  [Statistical] Full Sequence  ({f.tokens_evaluated}/{sc.total_tokens_generated}): "
                     f"Mean Prob={f.mean_token_probability:.4f} | Mean LogProb={f.mean_log_probability:.4f} | "
                     f"Geom Prob={f.geometric_probability:.4f}")
    else:
        lines.append(f"  [Statistical] Mean Token Prob (Sequence Proxy): {gen_output.mean_token_probability:.4f}")

    # Verbalized result
    if conf_result:
        if conf_result.is_valid:
            lines.append(f"  [Verbalized]  Self-Reported Confidence: {conf_result.confidence_percent:.1f}% "
                         f"(score={conf_result.confidence_score:.4f}, raw='{conf_result.raw_response}')")
        else:
            lines.append(f"  [Verbalized]  Self-Reported Confidence: FAILED ({conf_result.parsing_error}, raw='{conf_result.raw_response}')")
    else:
        lines.append("  [Verbalized]  Self-Reported Confidence: Not evaluated")

    lines.append("  * Note: Token probabilities are sequence-level proxies, NOT factual correctness probabilities.")
    return "\n".join(lines)


def format_comparison_display(comp: ComparisonResult) -> str:
    lines = []
    lines.append(f"  [Final Classification] : {comp.final_arbiter_classification}")
    lines.append(f"  [Decision Source]      : {comp.classification_decision_source}")
    if comp.arbiter_override_reason:
        lines.append(f"  [Arbiter Reason]       : {comp.arbiter_override_reason}")
    lines.append(f"  [Evidence Grounding]   : {comp.evidence_validation_status}")
    lines.append(f"  [Overall Rationale]    : {comp.overall_rationale}")
    
    # Claims summary
    lines.append(f"  [Claims Evaluated]     : Total={len(comp.claims)} | Matched={len(comp.matched_claims)} | "
                 f"Contradictory={len(comp.contradictory_claims)} | Ambiguous={len(comp.ambiguous_claims)}")

    if comp.contradictory_claims:
        lines.append("  [Contradictions Detected]:")
        for c in comp.contradictory_claims:
            lines.append(f"    - [{c.claim_id}] {c.interpreted_claim}")
            lines.append(f"      Original evidence : '{c.original_evidence_span}'")
            lines.append(f"      English evidence  : '{c.english_evidence_span}'")
            lines.append(f"      Explanation       : {c.explanation}")

    if comp.matched_claims:
        lines.append("  [Matched Claims]:")
        for c in comp.matched_claims:
            lines.append(f"    - [{c.claim_id}] {c.interpreted_claim}")
            lines.append(f"      Original evidence : '{c.original_evidence_span}' <==> English: '{c.english_evidence_span}'")

    if comp.original_only_claims:
        lines.append("  [Original-Only Details]:")
        for c in comp.original_only_claims:
            lines.append(f"    - [{c.claim_id}] {c.interpreted_claim} (Span: '{c.original_evidence_span}')")

    if comp.english_only_claims:
        lines.append("  [English-Only Details]:")
        for c in comp.english_only_claims:
            lines.append(f"    - [{c.claim_id}] {c.interpreted_claim} (Span: '{c.english_evidence_span}')")

    if comp.deterministic_features.candidate_discrepancies:
        lines.append("  [Candidate Deterministic Discrepancies]:")
        for d in comp.deterministic_features.candidate_discrepancies:
            lines.append(f"    - {d.category}: Orig='{d.original_value}' vs Eng='{d.english_value}'")

    lines.append("  * Note: Neither answer is ground truth. A CONSISTENT verdict denotes mutual factual agreement.")
    return "\n".join(lines)


def run_interactive_cli():
    print("=" * 80)
    print("CROSS-LINGUAL CALIBRATION ENGINE - PHASE 4 CLI PROTOTYPE")
    print("Model: Google Gemma 3 4B Instruct (4-bit NF4)")
    print("Features: Paired Answers, Dual Statistical Metrics, Strict Verbalized Confidence,")
    print("          Evidence-Grounded Factual Comparison Engine")
    print("Supported Languages: English (en), Kannada (kn), Hindi (hi),")
    print("                     Tamil (ta), Telugu (te), Malayalam (ml)")
    print("=" * 80)

    print("\n[INFO] Initializing pipeline and verifying model...")
    runner = ExperimentRunner()
    
    # Pre-load model once
    runner.model_engine.load_model()
    gpu_status = runner.model_engine.get_gpu_memory_status()
    print(f"[INFO] Model loaded successfully ({runner.model_engine.model_revision}). "
          f"GPU Allocated: {gpu_status['allocated_mb']:.1f} MB | Reserved: {gpu_status['reserved_mb']:.1f} MB")
    print("Type 'exit' or 'quit' at any prompt to terminate.\n")

    while True:
        try:
            print("-" * 80)
            question = input("\nEnter question: ").strip()
            if question.lower() in ("exit", "quit"):
                print("Exiting CLI.")
                break
            if not question:
                continue

            # Run detection
            detection = runner.detector.detect(question)
            
            if detection.is_uncertain:
                print(f"\n[ALERT] Language detection is uncertain ({detection.details.get('reason', 'low evidence')}).")
                print(f"Supported language codes: {list(SUPPORTED_LANGUAGES.keys())}")
                choice = input("Please specify the language code manually (or press Enter to skip): ").strip().lower()
                if choice in ("exit", "quit"):
                    break
                if choice in SUPPORTED_LANGUAGES:
                    detection = runner.detector.detect(question, override_lang=choice)
                else:
                    print("[SKIPPED] Cannot proceed with unverified language input.")
                    continue
            else:
                print(f"\nDetected Language: {detection.language_name} ({detection.language_code}) "
                      f"[Confidence: {detection.confidence:.2f} via {detection.detection_method}]")
                override = input(f"Press Enter to proceed with {detection.language_name}, or enter override code ({','.join(SUPPORTED_LANGUAGES.keys())}): ").strip().lower()
                if override in ("exit", "quit"):
                    break
                if override in SUPPORTED_LANGUAGES:
                    detection = runner.detector.detect(question, override_lang=override)
                    print(f"Manual override applied: {detection.language_name} ({detection.language_code})")

            print("\n[INFO] Generating paired answers, localized confidence, and comparison judge...")
            res = runner.run_pipeline(
                question=question,
                override_lang=detection.language_code if detection.is_overridden else None
            )

            if res["status"] != "SUCCESS":
                print(f"[ERROR] Generation aborted: {res['status']}")
                continue

            paired: PairedAnswerResult = res["paired_result"]
            orig = paired.original_answer
            eng = paired.english_answer
            comp: Optional[ComparisonResult] = res["comparison_result"]

            print("\n" + "=" * 80)
            print(f"1. ORIGINAL-LANGUAGE ANSWER ({paired.detected_language_name})")
            print("=" * 80)
            print(orig.answer_text)
            print("-" * 80)
            print(format_confidence_display(orig, paired.original_confidence))

            print("\n" + "=" * 80)
            print("2. INDEPENDENT ENGLISH ANSWER")
            print("=" * 80)
            print(eng.answer_text)
            print("-" * 80)
            print(format_confidence_display(eng, paired.english_confidence))

            if comp:
                print("\n" + "=" * 80)
                print("3. ANSWER COMPARISON & CONSISTENCY VERDICT")
                print("=" * 80)
                print(format_comparison_display(comp))

            print("\n" + "=" * 80)
            print(f"[SAVED] Run ID '{res['run_id']}' persisted to '{runner.output_path}'.")
            gpu_after = runner.model_engine.get_gpu_memory_status()
            print(f"GPU Memory: {gpu_after['allocated_mb']:.1f} MB allocated | Peak: {gpu_after['peak_mb']:.1f} MB")
            print("=" * 80)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    run_interactive_cli()
