"""
Phase 5 Full-Scale Empirical GPU Execution Engine.

Authoritative Implementation following Frozen Specification v4.5:
- Ingests 150 questions (data/full_150_questions.jsonl):
  * P_cal: 75 questions
  * P_val: 30 questions
  * P_test: 45 questions
- Evaluates 11 conditions per question:
  1,650 answer generations + 1,650 verbalized confidences + 750 comparisons = 4,050 GPU inferences.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (NF4 4-bit, Gemma 3 4B Instruct).
- Atomic incremental persistence with checkpoint recovery.
- Granular task invocation logging (results/empirical/model_call_logs.jsonl).
- Model selection on P_val strictly between Model 1 and Model 2 on Brier Score.
- Final evaluation and B=10,000 bootstrap tests strictly on P_test.
- Isolated output directory: results/empirical/
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output encoding on Windows console with line buffering
import io
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import torch
import numpy as np

from src.model_engine import ModelEngine
from src.confidence import ConfidenceEngine
from src.answer_comparator import AnswerComparator
from src.calibration.empirical_driver import EmpiricalPipelineDriver, CONDITIONS, LANG_NAMES
from src.calibration.instrumentation import ModelCallLogger
from src.calibration.schemas import CorrectnessEvaluator
from src.calibration.metrics import compute_brier_score, compute_quantile_ece, compute_murphy_decomposition
from src.calibration.eligibility import EligibilityEngine
from src.calibration.recalibration import RecalibrationEngine, Standardizer
from src.calibration.statistical_tests import HamiltonHareAllocator, BootstrapInference, MultiplicityController


# 5 Non-English conditions that are compared against English
JUDGE_TARGET_CONDITIONS = ["hi", "bn", "te", "ta", "hi_trans_en"]


def execute_full_scale_empirical_run(
    manifest_path: str = "data/full_150_questions.jsonl",
    output_dir: str = "results/empirical",
    seed: int = 20260330,
    progress_interval: int = 10,
) -> Dict[str, Any]:
    run_start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    records_file = os.path.join(output_dir, "phase5_empirical_records.jsonl")
    comparisons_file = os.path.join(output_dir, "phase5_empirical_comparisons.jsonl")
    calls_log_file = os.path.join(output_dir, "model_call_logs.jsonl")
    report_file = os.path.join(output_dir, "phase5_final_empirical_report.json")

    logger = ModelCallLogger(calls_log_file)

    print("=" * 75)
    print("PHASE 5: FULL-SCALE EMPIRICAL GPU EXECUTION (GEMMA 3 4B INSTRUCT)")
    print("=" * 75)
    print(f"Timestamp:             {datetime.now(timezone.utc).isoformat()}")
    print(f"Manifest Path:         {manifest_path}")
    print(f"Output Directory:      {output_dir}")
    print(f"Random Seed:           {seed}")
    print(f"CUDA Available:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:            {torch.cuda.get_device_name(0)}")
        print(f"Baseline VRAM:         {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")

    # 1. Load Question Manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    assert len(questions) == 150, f"Expected 150 questions, found {len(questions)}"
    q_map = {q["question_id"]: q for q in questions}

    # 2. Initialize Model Engine & Drivers
    model_engine = ModelEngine.get_instance()
    driver = EmpiricalPipelineDriver(
        output_dir=output_dir,
        model_engine=model_engine,
    )
    comparator = AnswerComparator(
        model_engine=model_engine,
    )

    # 3. Checkpoint Loading: Records
    completed_records = driver.load_completed_records()
    print(f"Checkpoint check: Loaded {len(completed_records)} existing answer records.")

    # Checkpoint Loading: Comparisons
    completed_comparisons: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(comparisons_file):
        with open(comparisons_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c_data = json.loads(line)
                    completed_comparisons[(c_data["question_id"], c_data["condition"])] = c_data
    print(f"Checkpoint check: Loaded {len(completed_comparisons)} existing judge comparisons.")

    # 4. Main Inference Loop: 1,650 Answer Records & Verbalized Confidences
    print("\n--- STAGE 1: ANSWER GENERATION & VERBALIZED CONFIDENCE ELICITATION ---")
    total_answers_expected = 150 * 11
    
    with open(records_file, "a", encoding="utf-8") as f_records:
        for q_idx, q in enumerate(questions, start=1):
            qid = q["question_id"]
            for cond in CONDITIONS:
                key = (qid, cond)
                if key in completed_records:
                    continue

                t0 = time.time()
                # Live GPU inference
                rec = driver.run_inference_for_record(q, cond)
                elapsed = time.time() - t0

                f_records.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_records.flush()
                completed_records[key] = rec

                # Log calls
                logger.record_call(
                    call_id=f"{rec['record_id']}_gen",
                    task_type="ANSWER_GENERATION",
                    question_id=qid,
                    condition=cond,
                    model_id=model_engine.model_id,
                    max_new_tokens=128,
                    completion_status="SUCCESS",
                    execution_time_seconds=elapsed,
                    output_preview=rec["answer_text"],
                )
                logger.record_call(
                    call_id=f"{rec['record_id']}_conf",
                    task_type="VERBALIZED_CONFIDENCE",
                    question_id=qid,
                    condition=cond,
                    model_id=model_engine.model_id,
                    max_new_tokens=10,
                    completion_status="SUCCESS" if rec["is_verbalized_valid"] else "PARSE_FAILURE",
                    execution_time_seconds=0.5,
                    output_preview=str(rec["verbalized_confidence_raw"]),
                )

                vram_now = torch.cuda.memory_allocated(0)/(1024**2) if torch.cuda.is_available() else 0.0
                print(f"  [Record {len(completed_records):4d}/{total_answers_expected}] ({qid}, {cond}) generated in {elapsed:.2f}s | VRAM: {vram_now:.1f} MB", flush=True)

                # Atomically update live progress file
                status_payload = {
                    "stage": "STAGE_1_ANSWER_GENERATION",
                    "completed_records": len(completed_records),
                    "total_records": total_answers_expected,
                    "last_record_id": rec["record_id"],
                    "question_id": qid,
                    "condition": cond,
                    "elapsed_seconds": round(elapsed, 2),
                    "vram_allocated_mb": round(vram_now, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(os.path.join(output_dir, "progress_status.json"), "w", encoding="utf-8") as f_st:
                    json.dump(status_payload, f_st, indent=2)

            if q_idx % progress_interval == 0 or q_idx == len(questions):
                vram_now = torch.cuda.memory_allocated(0)/(1024**2) if torch.cuda.is_available() else 0.0
                print(f"  [Checkpoint {q_idx:3d}/150 Questions] Answer records: {len(completed_records)}/{total_answers_expected} | Current VRAM: {vram_now:.2f} MB", flush=True)

    assert len(completed_records) == 1650, f"Expected 1,650 records, got {len(completed_records)}"
    print(f"Stage 1 Complete: 1,650 answer records generated and persisted.")

    # 5. Main Inference Loop: 750 Judge Inferences
    print("\n--- STAGE 2: CROSS-LINGUAL SYMMETRICAL COMPARISON JUDGE ---")
    total_judges_expected = 150 * len(JUDGE_TARGET_CONDITIONS)  # 150 * 5 = 750

    with open(comparisons_file, "a", encoding="utf-8") as f_comp:
        for q_idx, q in enumerate(questions, start=1):
            qid = q["question_id"]
            en_rec = completed_records.get((qid, "en"))
            assert en_rec is not None, f"Missing English record for {qid}"

            for cond in JUDGE_TARGET_CONDITIONS:
                c_key = (qid, cond)
                if c_key in completed_comparisons:
                    continue

                orig_rec = completed_records.get(c_key)
                assert orig_rec is not None, f"Missing target record for {c_key}"

                t0 = time.time()
                comp_res = comparator.compare(
                    question=q["text_en"],
                    original_answer_text=orig_rec["answer_text"],
                    english_answer_text=en_rec["answer_text"],
                    original_language_code=cond.split("_")[0],
                    original_language_name=LANG_NAMES.get(cond.split("_")[0], "Original"),
                    run_id=f"run_full_{qid}_{cond}",
                )
                elapsed = time.time() - t0

                comp_dict = {
                    "question_id": qid,
                    "condition": cond,
                    "comparison_id": comp_res.comparison_id,
                    "raw_judge_classification": comp_res.raw_judge_classification,
                    "final_arbiter_classification": comp_res.final_arbiter_classification,
                    "classification_decision_source": comp_res.classification_decision_source,
                    "evidence_validation_status": comp_res.evidence_validation_status,
                    "overall_rationale": comp_res.overall_rationale,
                    "claims_count": len(comp_res.claims),
                    "execution_time_seconds": elapsed,
                }
                f_comp.write(json.dumps(comp_dict, ensure_ascii=False) + "\n")
                f_comp.flush()
                completed_comparisons[c_key] = comp_dict

                logger.record_call(
                    call_id=comp_res.comparison_id,
                    task_type="JUDGE_INFERENCE",
                    question_id=qid,
                    condition=cond,
                    model_id=model_engine.model_id,
                    max_new_tokens=1024,
                    completion_status=comp_res.execution_status,
                    execution_time_seconds=elapsed,
                    output_preview=comp_res.overall_rationale,
                )

                vram_now = torch.cuda.memory_allocated(0)/(1024**2) if torch.cuda.is_available() else 0.0
                print(f"  [Judge {len(completed_comparisons):3d}/{total_judges_expected}] ({qid}, {cond}) evaluated in {elapsed:.2f}s | VRAM: {vram_now:.1f} MB", flush=True)

                status_payload = {
                    "stage": "STAGE_2_JUDGE_EVALUATION",
                    "completed_comparisons": len(completed_comparisons),
                    "total_comparisons": total_judges_expected,
                    "last_comparison_id": comp_res.comparison_id,
                    "question_id": qid,
                    "condition": cond,
                    "elapsed_seconds": round(elapsed, 2),
                    "vram_allocated_mb": round(vram_now, 1),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(os.path.join(output_dir, "progress_status.json"), "w", encoding="utf-8") as f_st:
                    json.dump(status_payload, f_st, indent=2)

            if q_idx % progress_interval == 0 or q_idx == len(questions):
                vram_now = torch.cuda.memory_allocated(0)/(1024**2) if torch.cuda.is_available() else 0.0
                print(f"  [Checkpoint {q_idx:3d}/150 Questions] Judge inferences: {len(completed_comparisons)}/{total_judges_expected} | Current VRAM: {vram_now:.2f} MB", flush=True)

    assert len(completed_comparisons) == 750, f"Expected 750 comparisons, got {len(completed_comparisons)}"
    print(f"Stage 2 Complete: 750 judge evaluations completed and persisted.")

    # 6. Empirical Analysis & Recalibration Pipeline
    print("\n--- STAGE 3: CALIBRATION FITTING, MODEL SELECTION & HELD-OUT EVALUATION ---")
    all_records = list(completed_records.values())

    cal_records = [r for r in all_records if r["partition"] == "cal"]
    val_records = [r for r in all_records if r["partition"] == "val"]
    test_records = [r for r in all_records if r["partition"] == "test"]

    assert len(cal_records) == 75 * 11, f"Expected 825 cal records, got {len(cal_records)}"
    assert len(val_records) == 30 * 11, f"Expected 330 val records, got {len(val_records)}"
    assert len(test_records) == 45 * 11, f"Expected 495 test records, got {len(test_records)}"

    all_qids = [q["question_id"] for q in questions]
    cal_qids = [q["question_id"] for q in questions if q["partition"] == "cal"]
    val_qids = [q["question_id"] for q in questions if q["partition"] == "val"]
    test_qids = [q["question_id"] for q in questions if q["partition"] == "test"]

    # Table A (Full-Dataset Attrition)
    elig_engine = EligibilityEngine(all_records)
    table_a = elig_engine.compute_table_a(all_qids)

    # Table B (Held-Out Test Paired Cohort Attrition)
    table_b_test = elig_engine.compute_table_b("P_test", test_qids, reference_condition="en")

    # Filter Dual-Track Eligible records
    cal_dual_ok = [
        r for r in cal_records
        if r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]
    ]
    val_dual_ok = [
        r for r in val_records
        if r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]
    ]
    test_dual_ok = [
        r for r in test_records
        if r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]
    ]

    # Recalibration Fitting on P_cal strictly
    c_v_cal = np.array([r["verbalized_confidence"] for r in cal_dual_ok], dtype=float)
    c_s_cal = np.array([r["statistical_confidence"] for r in cal_dual_ok], dtype=float)
    y_cal = np.array([r["strict_correctness"] for r in cal_dual_ok], dtype=float)
    conds_cal = [r["condition"] for r in cal_dual_ok]

    recal_verb = RecalibrationEngine()
    recal_verb.fit_on_calibration(c_v_cal, y_cal, conds_cal)

    recal_stat = RecalibrationEngine()
    recal_stat.fit_on_calibration(c_s_cal, y_cal, conds_cal)

    # Operational Model Selection on P_val strictly
    c_v_val = np.array([r["verbalized_confidence"] for r in val_dual_ok], dtype=float)
    c_s_val = np.array([r["statistical_confidence"] for r in val_dual_ok], dtype=float)
    y_val = np.array([r["strict_correctness"] for r in val_dual_ok], dtype=float)
    conds_val = [r["condition"] for r in val_dual_ok]

    best_model_verb = recal_verb.select_model_on_validation(c_v_val, y_val, conds_val)
    val_bs_verb = recal_verb.validation_brier_scores

    best_model_stat = recal_stat.select_model_on_validation(c_s_val, y_val, conds_val)
    val_bs_stat = recal_stat.validation_brier_scores

    print(f"Model Selection (Verbalized):  {best_model_verb} (Validation BS: {val_bs_verb})")
    print(f"Model Selection (Statistical): {best_model_stat} (Validation BS: {val_bs_stat})")

    # Evaluation on P_test strictly
    test_condition_metrics = {}
    for cond in CONDITIONS:
        cond_test = [r for r in test_dual_ok if r["condition"] == cond]
        if not cond_test:
            continue

        c_v_test = np.array([r["verbalized_confidence"] for r in cond_test], dtype=float)
        c_s_test = np.array([r["statistical_confidence"] for r in cond_test], dtype=float)
        y_test = np.array([r["strict_correctness"] for r in cond_test], dtype=float)

        c_v_recal = recal_verb.predict_recalibrated(c_v_test, [cond] * len(c_v_test))
        c_s_recal = recal_stat.predict_recalibrated(c_s_test, [cond] * len(c_s_test))

        uncal_v_m = compute_quantile_ece(c_v_test, y_test, num_bins=5)
        recal_v_m = compute_quantile_ece(c_v_recal, y_test, num_bins=5)

        uncal_s_m = compute_quantile_ece(c_s_test, y_test, num_bins=5)
        recal_s_m = compute_quantile_ece(c_s_recal, y_test, num_bins=5)

        test_condition_metrics[cond] = {
            "num_test_samples": len(cond_test),
            "verbalized": {
                "uncalibrated_bs": uncal_v_m.brier_score,
                "uncalibrated_ece": uncal_v_m.ece,
                "uncalibrated_mce": uncal_v_m.mce,
                "uncalibrated_rmsce": uncal_v_m.rmsce,
                "recalibrated_bs": recal_v_m.brier_score,
                "recalibrated_ece": recal_v_m.ece,
                "recalibrated_mce": recal_v_m.mce,
                "recalibrated_rmsce": recal_v_m.rmsce,
                "murphy_decomposition": {
                    "reliability": uncal_v_m.decomposition.reliability,
                    "resolution": uncal_v_m.decomposition.resolution,
                    "uncertainty": uncal_v_m.decomposition.uncertainty,
                    "residual": uncal_v_m.decomposition.bin_covariance_residual,
                },
            },
            "statistical": {
                "uncalibrated_bs": uncal_s_m.brier_score,
                "uncalibrated_ece": uncal_s_m.ece,
                "uncalibrated_mce": uncal_s_m.mce,
                "uncalibrated_rmsce": uncal_s_m.rmsce,
                "recalibrated_bs": recal_s_m.brier_score,
                "recalibrated_ece": recal_s_m.ece,
                "recalibrated_mce": recal_s_m.mce,
                "recalibrated_rmsce": recal_s_m.rmsce,
            },
        }

    # 7. Confirmatory Paired Bootstrap Hypothesis Testing (B = 10,000)
    print("\n--- STAGE 4: CONFIRMATORY PAIRED BOOTSTRAP HYPOTHESIS TESTING (B=10,000) ---")
    bootstrap_results = {}
    target_languages = ["hi", "bn", "te", "ta"]

    for lang in target_languages:
        def delta_ece_lang(qids_sub: List[str]) -> float:
            sub_set = set(qids_sub)
            lang_records = [r for r in test_dual_ok if r["condition"] == lang and r["question_id"] in sub_set]
            en_records = [r for r in test_dual_ok if r["condition"] == "en" and r["question_id"] in sub_set]
            if not lang_records or not en_records:
                return 0.0

            c_lang = np.array([r["verbalized_confidence"] for r in lang_records])
            y_lang = np.array([r["strict_correctness"] for r in lang_records])
            c_en = np.array([r["verbalized_confidence"] for r in en_records])
            y_en = np.array([r["strict_correctness"] for r in en_records])

            ece_lang = compute_quantile_ece(c_lang, y_lang, num_bins=3).ece
            ece_en = compute_quantile_ece(c_en, y_en, num_bins=3).ece
            return ece_lang - ece_en

        res_a = BootstrapInference.test_paired_difference(
            test_qids, delta_ece_lang, fwer_tier="Level_A_Within_Track", n_bootstraps=10000, seed=seed
        )
        res_b = BootstrapInference.test_paired_difference(
            test_qids, delta_ece_lang, fwer_tier="Level_B_Global", n_bootstraps=10000, seed=seed
        )
        bootstrap_results[lang] = {
            "level_a": res_a.__dict__,
            "level_b": res_b.__dict__,
        }
        print(f"  [{lang} vs en] Level A 99% CI: [{res_a.ci_lower:+.4f}, {res_a.ci_upper:+.4f}] (Significant: {res_a.is_significant})")
        print(f"  [{lang} vs en] Level B 99.75% CI: [{res_b.ci_lower:+.4f}, {res_b.ci_upper:+.4f}] (Significant: {res_b.is_significant})")

    # 8. 10% Stratified Random QC Audit
    all_claims_by_cond: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        cond = r["condition"]
        if cond not in all_claims_by_cond:
            all_claims_by_cond[cond] = []
        for c in r["claims"]:
            all_claims_by_cond[cond].append(c)

    total_claims = sum(len(c) for c in all_claims_by_cond.values())
    target_qc_size = math.ceil(total_claims * 0.10)
    qc_sampled = HamiltonHareAllocator.sample_strata_indices(
        all_claims_by_cond, target_qc_size, seed=seed
    )
    total_qc_sampled = sum(len(v) for v in qc_sampled.values())

    qc_manifest_file = os.path.join(output_dir, "phase5_qc_audit_manifest.json")
    with open(qc_manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_claims": total_claims,
            "target_qc_size": target_qc_size,
            "actual_qc_sampled": total_qc_sampled,
            "qc_sampled_by_condition": {k: len(v) for k, v in qc_sampled.items()},
            "seed": seed,
        }, f, indent=2)

    total_duration = time.time() - run_start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(0)/(1024**2) if torch.cuda.is_available() else 0.0

    call_summary = logger.summary()

    # 9. Final Comprehensive Report
    report = {
        "status": "COMPLETED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(total_duration, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_id": model_engine.model_id,
            "model_revision": model_engine.model_revision,
        },
        "questions_total": len(questions),
        "total_records": len(all_records),
        "total_comparisons": len(completed_comparisons),
        "total_claims": total_claims,
        "qc_audit_sampled": total_qc_sampled,
        "partitions": {
            "P_cal_records": len(cal_records),
            "P_val_records": len(val_records),
            "P_test_records": len(test_records),
        },
        "model_invocations": call_summary,
        "selected_operational_model": {
            "verbalized": best_model_verb,
            "verbalized_validation_brier_scores": val_bs_verb,
            "statistical": best_model_stat,
            "statistical_validation_brier_scores": val_bs_stat,
        },
        "recalibration_parameters": {
            "model_1": {"w": recal_verb.model_1.w, "b": recal_verb.model_1.b},
            "model_2": {"w": recal_verb.model_2.w, "b0": recal_verb.model_2.b0, "deltas": recal_verb.model_2.deltas},
            "model_3_exploratory_languages": len(recal_verb.model_3.models),
        },
        "held_out_test_condition_metrics": test_condition_metrics,
        "confirmatory_bootstrap_hypothesis_tests": bootstrap_results,
        "table_a_full_attrition": [r.__dict__ for r in table_a],
        "table_b_test_attrition": [r.__dict__ for r in table_b_test],
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"FULL-SCALE EMPIRICAL GPU EXECUTION COMPLETED in {total_duration:.2f}s")
    print(f"Total Model Invocations Logged: {call_summary['TOTAL']}")
    print(f"Final Report Written to:        {report_file}")
    print("=" * 75)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Full-Scale Empirical GPU Execution")
    parser.add_argument("--manifest", type=str, default="data/full_150_questions.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/empirical")
    parser.add_argument("--seed", type=int, default=20260330)
    args = parser.parse_args()

    execute_full_scale_empirical_run(args.manifest, args.output_dir, args.seed)
