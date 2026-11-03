"""
Phase 5 Empirical Pipeline Orchestrator.

Integrates the empirical driver with checkpoint/resume capability,
strict B = 10,000 bootstrap compliance, Table B exact conjunctions,
and persistent empirical artifact generation in results/empirical/.
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

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.calibration.empirical_driver import EmpiricalPipelineDriver, CONDITIONS
from src.calibration.metrics import compute_brier_score, compute_quantile_ece, compute_murphy_decomposition
from src.calibration.eligibility import EligibilityEngine
from src.calibration.recalibration import RecalibrationEngine, Standardizer
from src.calibration.statistical_tests import HamiltonHareAllocator, BootstrapInference, MultiplicityController


def execute_empirical_pipeline(
    manifest_path: str = "data/full_150_questions.jsonl",
    output_dir: str = "results/empirical",
    seed: int = 20260330,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes the genuine Phase 5 empirical pipeline against the Gemma 3 4B Instruct GPU model.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load 150-question manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if max_questions is not None:
        questions = questions[:max_questions]
        print(f"Executing scoped empirical run: {len(questions)} questions.")

    driver = EmpiricalPipelineDriver(output_dir=output_dir)
    completed_records = driver.load_completed_records()
    print(f"Loaded {len(completed_records)} existing empirical records from checkpoint.")

    records_file = os.path.join(output_dir, "phase5_empirical_records.jsonl")
    all_records: List[Dict[str, Any]] = list(completed_records.values())

    # 2. Sequential Inference Loop with Atomic Checkpointing
    with open(records_file, "a", encoding="utf-8") as f_out:
        for q_idx, q in enumerate(questions, start=1):
            qid = q["question_id"]
            for cond in CONDITIONS:
                key = (qid, cond)
                if key in completed_records:
                    continue

                # Execute genuine model inference
                record = driver.run_inference_for_record(
                    question_data=q,
                    condition=cond,
                )
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                completed_records[key] = record
                all_records.append(record)

    total_records = len(all_records)
    print(f"Empirical records complete: {total_records} records in {records_file}")

    # 3. Partitions Separation
    cal_records = [r for r in all_records if r["partition"] == "cal"]
    val_records = [r for r in all_records if r["partition"] == "val"]
    test_records = [r for r in all_records if r["partition"] == "test"]

    # 4. Eligibility & Attrition Accounting
    all_qids = [q["question_id"] for q in questions]
    cal_qids = [q["question_id"] for q in questions if q["partition"] == "cal"]
    val_qids = [q["question_id"] for q in questions if q["partition"] == "val"]
    test_qids = [q["question_id"] for q in questions if q["partition"] == "test"]

    elig_engine = EligibilityEngine(all_records)
    table_a = elig_engine.compute_table_a(all_qids)

    table_b_cal = elig_engine.compute_table_b("P_cal", cal_qids, reference_condition="en")
    table_b_val = elig_engine.compute_table_b("P_val", val_qids, reference_condition="en")
    table_b_test = elig_engine.compute_table_b("P_test", test_qids, reference_condition="en")

    # 5. Calibration Fitting (P_cal only) & Selection (P_val only)
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

    c_v_cal = np.array([r["verbalized_confidence"] for r in cal_dual_ok], dtype=float)
    c_s_cal = np.array([r["statistical_confidence"] for r in cal_dual_ok], dtype=float)
    y_cal = np.array([r["strict_correctness"] for r in cal_dual_ok], dtype=float)
    conds_cal = [r["condition"] for r in cal_dual_ok]

    recal_verb = RecalibrationEngine()
    recal_stat = RecalibrationEngine()

    if len(c_v_cal) > 0:
        recal_verb.fit_on_calibration(c_v_cal, y_cal, conds_cal)
        recal_stat.fit_on_calibration(c_s_cal, y_cal, conds_cal)

    c_v_val = np.array([r["verbalized_confidence"] for r in val_dual_ok], dtype=float)
    c_s_val = np.array([r["statistical_confidence"] for r in val_dual_ok], dtype=float)
    y_val = np.array([r["strict_correctness"] for r in val_dual_ok], dtype=float)
    conds_val = [r["condition"] for r in val_dual_ok]

    best_model_verb = "Model_1_Pooled_Platt"
    best_model_stat = "Model_1_Pooled_Platt"
    if len(c_v_val) > 0:
        best_model_verb = recal_verb.select_model_on_validation(c_v_val, y_val, conds_val)
        best_model_stat = recal_stat.select_model_on_validation(c_s_val, y_val, conds_val)

    # 6. Held-Out Test Evaluation (P_test only)
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

        uncal_v_m = compute_quantile_ece(c_v_test, y_test, num_bins=min(5, len(cond_test)))
        recal_v_m = compute_quantile_ece(c_v_recal, y_test, num_bins=min(5, len(cond_test)))

        uncal_s_m = compute_quantile_ece(c_s_test, y_test, num_bins=min(5, len(cond_test)))
        recal_s_m = compute_quantile_ece(c_s_recal, y_test, num_bins=min(5, len(cond_test)))

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
    bootstrap_results = {}
    target_languages = ["hi", "bn", "te", "ta"]

    if len(test_dual_ok) > 0:
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

                ece_lang = compute_quantile_ece(c_lang, y_lang, num_bins=min(3, len(c_lang))).ece
                ece_en = compute_quantile_ece(c_en, y_en, num_bins=min(3, len(c_en))).ece
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

    runtime_sec = time.time() - start_time

    # 8. Compile Report
    report = {
        "status": "COMPLETED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime_sec, 2),
        "questions_total": len(questions),
        "total_records": total_records,
        "partitions": {
            "P_cal_records": len(cal_records),
            "P_val_records": len(val_records),
            "P_test_records": len(test_records),
        },
        "selected_operational_model": {
            "verbalized": best_model_verb,
            "statistical": best_model_stat,
        },
        "held_out_test_condition_metrics": test_condition_metrics,
        "confirmatory_bootstrap_hypothesis_tests": bootstrap_results,
        "table_a_full_attrition": [r.__dict__ for r in table_a],
        "table_b_test_attrition": [r.__dict__ for r in table_b_test],
    }

    report_path = os.path.join(output_dir, "phase5_empirical_execution_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Empirical Pipeline Orchestrator")
    parser.add_argument("--manifest", type=str, default="data/full_150_questions.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/empirical")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    execute_empirical_pipeline(args.manifest, args.output_dir, args.seed, args.max_questions)
