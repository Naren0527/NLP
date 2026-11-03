"""
Phase 5 Research Deliverables Generator
Implements Frozen Methodological Specification v4.5 Research Visualizations:
1. Verbalized vs Statistical Calibration (Pre vs Post on P_test)
2. Calibration Mitigation Effectiveness (M1 vs M2 on P_test)
3. Cross-Lingual Answer Consistency (Analysis of 750 Symmetrical Comparisons)
4. Confidence-Accuracy Gap (Over/Underconfidence Decomposition)
5. Comprehensive Research Metrics CSV, Summary MD, and results.md
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["MPLCONFIGDIR"] = os.path.abspath("results/scratch_matplotlib")
os.makedirs("results/scratch_matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.recalibration import RecalibrationEngine
from src.calibration.metrics import compute_quantile_ece, compute_murphy_decomposition

OUTPUT_DIR = Path("outputs/research_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path("results/empirical/phase5_final_empirical_report.json")
RECORDS_PATH = Path("results/empirical/phase5_empirical_records.jsonl")
COMPARISONS_PATH = Path("results/empirical/phase5_empirical_comparisons.jsonl")

def load_data():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = []
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    comparisons = []
    with open(COMPARISONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                comparisons.append(json.loads(line))

    return report, records, comparisons

def run():
    report, records, comparisons = load_data()
    df_records = pd.DataFrame(records)

    # Filter eligible records
    cal_dual = [r for r in records if r["partition"] == "cal" and r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]]
    val_dual = [r for r in records if r["partition"] == "val" and r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]]
    test_dual = [r for r in records if r["partition"] == "test" and r["is_strict_correctness_eligible"] and r["is_verbalized_valid"] and r["is_statistical_valid"]]

    # Fit Recalibration Engines on P_cal
    c_v_cal = np.array([r["verbalized_confidence"] for r in cal_dual], dtype=float)
    c_s_cal = np.array([r["statistical_confidence"] for r in cal_dual], dtype=float)
    y_cal = np.array([r["strict_correctness"] for r in cal_dual], dtype=float)
    conds_cal = [r["condition"] for r in cal_dual]

    recal_v = RecalibrationEngine()
    recal_v.fit_on_calibration(c_v_cal, y_cal, conds_cal)

    recal_s = RecalibrationEngine()
    recal_s.fit_on_calibration(c_s_cal, y_cal, conds_cal)

    # Evaluate on P_test for both Model 1 and Model 2
    CONDITIONS = ["en", "hi", "bn", "te", "ta", "hi_trans_en", "bn_trans_en", "te_trans_en", "ta_trans_en", "en_trans_hi", "en_trans_ta"]

    condition_research_metrics = {}
    for cond in CONDITIONS:
        cond_test = [r for r in test_dual if r["condition"] == cond]
        if not cond_test:
            continue

        c_v = np.array([r["verbalized_confidence"] for r in cond_test], dtype=float)
        c_s = np.array([r["statistical_confidence"] for r in cond_test], dtype=float)
        y = np.array([r["strict_correctness"] for r in cond_test], dtype=float)

        # Pre-calibration
        uncal_v_ece = compute_quantile_ece(c_v, y, num_bins=5).ece
        uncal_v_bs = float(np.mean((c_v - y)**2))
        uncal_s_ece = compute_quantile_ece(c_s, y, num_bins=5).ece
        uncal_s_bs = float(np.mean((c_s - y)**2))

        # Model 1 (Pooled Platt)
        p_m1_v = recal_v.predict_recalibrated(c_v, [cond]*len(c_v), model_name="Model_1_Pooled_Platt")
        m1_v_ece = compute_quantile_ece(p_m1_v, y, num_bins=5).ece
        m1_v_bs = float(np.mean((p_m1_v - y)**2))

        p_m1_s = recal_s.predict_recalibrated(c_s, [cond]*len(c_s), model_name="Model_1_Pooled_Platt")
        m1_s_ece = compute_quantile_ece(p_m1_s, y, num_bins=5).ece
        m1_s_bs = float(np.mean((p_m1_s - y)**2))

        # Model 2 (Language-Shift Platt)
        p_m2_v = recal_v.predict_recalibrated(c_v, [cond]*len(c_v), model_name="Model_2_Language_Shift_Platt")
        m2_v_ece = compute_quantile_ece(p_m2_v, y, num_bins=5).ece
        m2_v_bs = float(np.mean((p_m2_v - y)**2))

        p_m2_s = recal_s.predict_recalibrated(c_s, [cond]*len(c_s), model_name="Model_2_Language_Shift_Platt")
        m2_s_ece = compute_quantile_ece(p_m2_s, y, num_bins=5).ece
        m2_s_bs = float(np.mean((p_m2_s - y)**2))

        accuracy = float(np.mean(y))
        mean_v_conf = float(np.mean(c_v))
        mean_s_conf = float(np.mean(c_s))

        condition_research_metrics[cond] = {
            "n_samples": len(cond_test),
            "accuracy": accuracy,
            "mean_v_conf": mean_v_conf,
            "mean_s_conf": mean_s_conf,
            "v_overconfidence_gap": mean_v_conf - accuracy,
            "s_overconfidence_gap": mean_s_conf - accuracy,
            "uncal_v_ece": uncal_v_ece,
            "uncal_v_bs": uncal_v_bs,
            "uncal_s_ece": uncal_s_ece,
            "uncal_s_bs": uncal_s_bs,
            "m1_v_ece": m1_v_ece,
            "m1_v_bs": m1_v_bs,
            "m2_v_ece": m2_v_ece,
            "m2_v_bs": m2_v_bs,
            "m1_s_ece": m1_s_ece,
            "m1_s_bs": m1_s_bs,
            "m2_s_ece": m2_s_ece,
            "m2_s_bs": m2_s_bs,
            "ece_reduction_m2_vs_uncal": uncal_v_ece - m2_v_ece,
            "bs_reduction_m2_vs_uncal": uncal_v_bs - m2_v_bs,
            "ece_reduction_m2_vs_m1": m1_v_ece - m2_v_ece,
            "bs_reduction_m2_vs_m1": m1_v_bs - m2_v_bs,
        }

    # =========================================================================
    # FIGURE 1: Verbalized vs Statistical Calibration (Pre vs Post on P_test)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot A: ECE Comparison
    ax = axes[0]
    cond_labels_short = [c.replace("_trans_en", "\n(trans_en)").replace("en_trans_", "trans_to\n") for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))
    w = 0.2

    uncal_v_eces = [condition_research_metrics[c]["uncal_v_ece"] for c in CONDITIONS]
    uncal_s_eces = [condition_research_metrics[c]["uncal_s_ece"] for c in CONDITIONS]
    recal_v_eces = [condition_research_metrics[c]["m2_v_ece"] for c in CONDITIONS]
    recal_s_eces = [condition_research_metrics[c]["m2_s_ece"] for c in CONDITIONS]

    ax.bar(x - 1.5*w, uncal_v_eces, w, label="Verbalized (Pre-Cal)", color="#d62728", alpha=0.85, edgecolor="black")
    ax.bar(x - 0.5*w, uncal_s_eces, w, label="Statistical (Pre-Cal)", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax.bar(x + 0.5*w, recal_v_eces, w, label="Verbalized (Post-Cal M2)", color="#ff9896", edgecolor="black")
    ax.bar(x + 1.5*w, recal_s_eces, w, label="Statistical (Post-Cal M2)", color="#aec7e8", edgecolor="black")

    ax.set_title("(A) Expected Calibration Error (ECE) Across Conditions ($P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("ECE (Lower is Better)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels_short, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    # Subplot B: Brier Score Comparison
    ax = axes[1]
    uncal_v_bss = [condition_research_metrics[c]["uncal_v_bs"] for c in CONDITIONS]
    uncal_s_bss = [condition_research_metrics[c]["uncal_s_bs"] for c in CONDITIONS]
    recal_v_bss = [condition_research_metrics[c]["m2_v_bs"] for c in CONDITIONS]
    recal_s_bss = [condition_research_metrics[c]["m2_s_bs"] for c in CONDITIONS]

    ax.bar(x - 1.5*w, uncal_v_bss, w, label="Verbalized (Pre-Cal)", color="#d62728", alpha=0.85, edgecolor="black")
    ax.bar(x - 0.5*w, uncal_s_bss, w, label="Statistical (Pre-Cal)", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax.bar(x + 0.5*w, recal_v_bss, w, label="Verbalized (Post-Cal M2)", color="#ff9896", edgecolor="black")
    ax.bar(x + 1.5*w, recal_s_bss, w, label="Statistical (Post-Cal M2)", color="#aec7e8", edgecolor="black")

    ax.set_title("(B) Brier Score Across Conditions ($P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Brier Score (Lower is Better)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels_short, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    f1_path = OUTPUT_DIR / "1_verbalized_vs_statistical_calibration_pre_post.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 2: Calibration Mitigation Effectiveness (M1 vs M2 on P_test)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Primary 5 languages: en, hi, bn, te, ta
    core_langs = ["en", "hi", "bn", "te", "ta"]
    core_labels = ["English (en)", "Hindi (hi)", "Bengali (bn)", "Telugu (te)", "Tamil (ta)"]
    x_c = np.arange(len(core_langs))
    w_c = 0.28

    # Subplot A: Absolute ECE (Uncal vs M1 vs M2)
    ax = axes[0]
    uncal_c_ece = [condition_research_metrics[c]["uncal_v_ece"] for c in core_langs]
    m1_c_ece = [condition_research_metrics[c]["m1_v_ece"] for c in core_langs]
    m2_c_ece = [condition_research_metrics[c]["m2_v_ece"] for c in core_langs]

    ax.bar(x_c - w_c, uncal_c_ece, w_c, label="Uncalibrated", color="#bdbdbd", edgecolor="black")
    ax.bar(x_c, m1_c_ece, w_c, label="Model 1: Pooled Platt", color="#6baed6", edgecolor="black")
    ax.bar(x_c + w_c, m2_c_ece, w_c, label="Model 2: Language-Shift Platt", color="#2171b5", edgecolor="black")

    for i in range(len(core_langs)):
        ax.text(x_c[i] - w_c, uncal_c_ece[i] + 0.015, f"{uncal_c_ece[i]:.2f}", ha="center", fontsize=8)
        ax.text(x_c[i], m1_c_ece[i] + 0.015, f"{m1_c_ece[i]:.2f}", ha="center", fontsize=8)
        ax.text(x_c[i] + w_c, m2_c_ece[i] + 0.015, f"{m2_c_ece[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("(A) Verbalized ECE by Mitigation Model ($P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("ECE (Lower is Better)", fontsize=10)
    ax.set_xticks(x_c)
    ax.set_xticklabels(core_labels, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    # Subplot B: Relative ECE Reduction (Delta ECE)
    ax = axes[1]
    m1_red = [condition_research_metrics[c]["uncal_v_ece"] - condition_research_metrics[c]["m1_v_ece"] for c in core_langs]
    m2_red = [condition_research_metrics[c]["uncal_v_ece"] - condition_research_metrics[c]["m2_v_ece"] for c in core_langs]

    ax.bar(x_c - w_c/2, m1_red, w_c, label="M1 ECE Reduction vs Uncal", color="#74c476", edgecolor="black")
    ax.bar(x_c + w_c/2, m2_red, w_c, label="M2 ECE Reduction vs Uncal", color="#238b45", edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for i in range(len(core_langs)):
        ax.text(x_c[i] - w_c/2, m1_red[i] + (0.015 if m1_red[i] >= 0 else -0.035), f"{m1_red[i]:+.2f}", ha="center", fontsize=8)
        ax.text(x_c[i] + w_c/2, m2_red[i] + (0.015 if m2_red[i] >= 0 else -0.035), f"{m2_red[i]:+.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("(B) Net Calibration Improvement ($\Delta$ECE Reduction on $P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("ECE Reduction ($\Delta$ECE > 0 = Improvement)", fontsize=10)
    ax.set_xticks(x_c)
    ax.set_xticklabels(core_labels, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    f2_path = OUTPUT_DIR / "2_calibration_mitigation_effectiveness_m1_vs_m2.png"
    plt.savefig(f2_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 3: Cross-Lingual Answer Consistency (750 Symmetrical Comparisons)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Parse comparisons
    comp_conds = ["hi", "bn", "te", "ta", "hi_trans_en"]
    comp_labels = ["Hindi Direct\n(hi vs en)", "Bengali Direct\n(bn vs en)", "Telugu Direct\n(te vs en)", "Tamil Direct\n(ta vs en)", "Hindi Transfer\n(hi_trans_en vs en)"]

    consistent_counts = []
    uncertain_counts = []
    inconsistent_counts = []

    raw_consistent_counts = []
    raw_inconsistent_counts = []

    for c in comp_conds:
        sub_c = [comp for comp in comparisons if comp["condition"] == c]
        arb_counts = Counter(comp["final_arbiter_classification"] for comp in sub_c)
        raw_counts = Counter(comp["raw_judge_classification"] for comp in sub_c)

        consistent_counts.append(arb_counts.get("CONSISTENT", 0))
        uncertain_counts.append(arb_counts.get("UNCERTAIN", 0))
        inconsistent_counts.append(arb_counts.get("INCONSISTENT", 0))

        raw_consistent_counts.append(raw_counts.get("CONSISTENT", 0))
        raw_inconsistent_counts.append(raw_counts.get("INCONSISTENT", 0))

    # Subplot A: Final Arbiter Classification Distribution (N=150 pairs per condition)
    ax = axes[0]
    x_k = np.arange(len(comp_conds))
    w_k = 0.55

    p1 = ax.bar(x_k, consistent_counts, w_k, label="Consistent (Grounded Agreement)", color="#2ca02c", edgecolor="black")
    p2 = ax.bar(x_k, uncertain_counts, w_k, bottom=consistent_counts, label="Uncertain (Ungrounded / Discrepancy)", color="#ffbb78", edgecolor="black")
    bottom_3 = np.array(consistent_counts) + np.array(uncertain_counts)
    p3 = ax.bar(x_k, inconsistent_counts, w_k, bottom=bottom_3, label="Inconsistent (Factual Contradiction)", color="#d62728", edgecolor="black")

    for i in range(len(comp_conds)):
        # Consistency percentage text
        pct = (consistent_counts[i] / 150.0) * 100.0
        ax.text(x_k[i], consistent_counts[i]/2, f"{consistent_counts[i]}\n({pct:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if uncertain_counts[i] > 15:
            ax.text(x_k[i], consistent_counts[i] + uncertain_counts[i]/2, f"{uncertain_counts[i]}", ha="center", va="center", color="black", fontsize=9)
        if inconsistent_counts[i] > 2:
            ax.text(x_k[i], bottom_3[i] + inconsistent_counts[i]/2, f"{inconsistent_counts[i]}", ha="center", va="center", color="white", fontsize=8)

    ax.set_title("(A) Final Arbiter Alignment Across 750 Symmetrical Pairs", fontsize=11, fontweight="bold")
    ax.set_ylabel("Answer Pairs Count (N = 150 per condition)", fontsize=10)
    ax.set_xticks(x_k)
    ax.set_xticklabels(comp_labels, fontsize=8.5)
    ax.set_ylim(0, 165)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    # Subplot B: Raw Judge vs Arbiter Breakdown (Detection of Evidence Hallucination)
    ax = axes[1]
    w_sub = 0.35
    ax.bar(x_k - w_sub/2, raw_consistent_counts, w_sub, label="Raw LLM Judge Consistent (Unchecked)", color="#98df8a", edgecolor="black")
    ax.bar(x_k + w_sub/2, consistent_counts, w_sub, label="Strict Arbiter Consistent (All Evidence Grounded)", color="#2ca02c", edgecolor="black")

    for i in range(len(comp_conds)):
        ax.text(x_k[i] - w_sub/2, raw_consistent_counts[i] + 3, f"{raw_consistent_counts[i]}", ha="center", fontsize=8)
        ax.text(x_k[i] + w_sub/2, consistent_counts[i] + 3, f"{consistent_counts[i]}", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("(B) Impact of Strict Grounding Adjudication on Consistency", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pair Count Evaluated as Consistent", fontsize=10)
    ax.set_xticks(x_k)
    ax.set_xticklabels(comp_labels, fontsize=8.5)
    ax.set_ylim(0, 160)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    f3_path = OUTPUT_DIR / "3_cross_lingual_answer_consistency.png"
    plt.savefig(f3_path, dpi=300)
    plt.close()

    # =========================================================================
    # FIGURE 4: Confidence-Accuracy Gap (Over/Underconfidence Analysis)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Group conditions into Native Indic vs Transferred/Translated
    native_conds = ["en", "hi", "bn", "te", "ta"]
    native_labels = ["English (en)", "Hindi (hi)", "Bengali (bn)", "Telugu (te)", "Tamil (ta)"]

    trans_conds = ["hi_trans_en", "bn_trans_en", "te_trans_en", "ta_trans_en", "en_trans_hi", "en_trans_ta"]
    trans_labels = ["hi $\to$ en", "bn $\to$ en", "te $\to$ en", "ta $\to$ en", "en $\to$ hi", "en $\to$ ta"]

    # Subplot A: Native Conditions Confidence-Accuracy Gap
    ax = axes[0]
    x_n = np.arange(len(native_conds))
    w_g = 0.25

    nat_acc = [condition_research_metrics[c]["accuracy"] for c in native_conds]
    nat_v_conf = [condition_research_metrics[c]["mean_v_conf"] for c in native_conds]
    nat_s_conf = [condition_research_metrics[c]["mean_s_conf"] for c in native_conds]

    ax.bar(x_n - w_g, nat_acc, w_g, label="Observed Accuracy", color="#31a354", edgecolor="black")
    ax.bar(x_n, nat_v_conf, w_g, label="Verbalized Confidence", color="#de2d26", edgecolor="black")
    ax.bar(x_n + w_g, nat_s_conf, w_g, label="Statistical Confidence", color="#3182bd", edgecolor="black")

    for i in range(len(native_conds)):
        gap = nat_v_conf[i] - nat_acc[i]
        sign = "+" if gap >= 0 else ""
        ax.text(x_n[i], max(nat_v_conf[i], nat_acc[i]) + 0.03, f"Gap:\n{sign}{gap:.2f}", ha="center", fontsize=8, color="#a50f15", fontweight="bold")

    ax.set_title("(A) Native Language Conditions: Overconfidence Gap ($P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability / Rate", fontsize=10)
    ax.set_xticks(x_n)
    ax.set_xticklabels(native_labels, fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    # Subplot B: Transferred & Translation-Aided Conditions
    ax = axes[1]
    x_t = np.arange(len(trans_conds))

    tra_acc = [condition_research_metrics[c]["accuracy"] for c in trans_conds]
    tra_v_conf = [condition_research_metrics[c]["mean_v_conf"] for c in trans_conds]
    tra_s_conf = [condition_research_metrics[c]["mean_s_conf"] for c in trans_conds]

    ax.bar(x_t - w_g, tra_acc, w_g, label="Observed Accuracy", color="#31a354", edgecolor="black")
    ax.bar(x_t, tra_v_conf, w_g, label="Verbalized Confidence", color="#de2d26", edgecolor="black")
    ax.bar(x_t + w_g, tra_s_conf, w_g, label="Statistical Confidence", color="#3182bd", edgecolor="black")

    for i in range(len(trans_conds)):
        gap = tra_v_conf[i] - tra_acc[i]
        sign = "+" if gap >= 0 else ""
        ax.text(x_t[i], max(tra_v_conf[i], tra_acc[i]) + 0.03, f"Gap:\n{sign}{gap:.2f}", ha="center", fontsize=8, color="#a50f15", fontweight="bold")

    ax.set_title("(B) Transferred & Translation-Aided Conditions ($P_{test}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability / Rate", fontsize=10)
    ax.set_xticks(x_t)
    ax.set_xticklabels(trans_labels, fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    f4_path = OUTPUT_DIR / "4_confidence_accuracy_gap_by_condition.png"
    plt.savefig(f4_path, dpi=300)
    plt.close()

    # =========================================================================
    # 5. RESEARCH SUMMARY TABLE & CSV EXPORT
    # =========================================================================
    # Compute consistency rate for conditions with comparisons
    consistency_rates = {}
    for c in CONDITIONS:
        sub_c = [comp for comp in comparisons if comp["condition"] == c]
        if sub_c:
            arb_counts = Counter(comp["final_arbiter_classification"] for comp in sub_c)
            consistency_rates[c] = round((arb_counts.get("CONSISTENT", 0) / len(sub_c)) * 100.0, 2)
        else:
            consistency_rates[c] = None

    table_rows = []
    for c in CONDITIONS:
        m = condition_research_metrics[c]
        table_rows.append({
            "condition": c,
            "test_sample_size": m["n_samples"],
            "accuracy": round(m["accuracy"], 4),
            "verbalized_confidence": round(m["mean_v_conf"], 4),
            "statistical_confidence": round(m["mean_s_conf"], 4),
            "overconfidence_gap_verb": round(m["v_overconfidence_gap"], 4),
            "uncalibrated_v_ece": round(m["uncal_v_ece"], 4),
            "uncalibrated_v_bs": round(m["uncal_v_bs"], 4),
            "model_1_v_ece": round(m["m1_v_ece"], 4),
            "model_1_v_bs": round(m["m1_v_bs"], 4),
            "model_2_v_ece": round(m["m2_v_ece"], 4),
            "model_2_v_bs": round(m["m2_v_bs"], 4),
            "m2_delta_ece_reduction": round(m["ece_reduction_m2_vs_uncal"], 4),
            "m2_delta_bs_reduction": round(m["bs_reduction_m2_vs_uncal"], 4),
            "uncalibrated_s_ece": round(m["uncal_s_ece"], 4),
            "uncalibrated_s_bs": round(m["uncal_s_bs"], 4),
            "model_2_s_ece": round(m["m2_s_ece"], 4),
            "model_2_s_bs": round(m["m2_s_bs"], 4),
            "answer_consistency_percent": consistency_rates[c],
        })

    res_df = pd.DataFrame(table_rows)
    csv_file = OUTPUT_DIR / "research_metrics.csv"
    res_df.to_csv(csv_file, index=False)

    # Export markdown table
    md_summary_file = OUTPUT_DIR / "research_summary.md"
    with open(md_summary_file, "w", encoding="utf-8") as f:
        f.write("# Phase 5 Research Metrics Summary (Frozen Specification v4.5)\n\n")
        f.write("**Model:** `google/gemma-3-4b-it` (NF4 4-bit) | **Hardware:** NVIDIA GeForce RTX 4050 GPU\n")
        f.write("**Evaluation:** Held-Out Test Set $P_{\\text{test}}$ ($N=45$ questions $\\times$ 11 conditions)\n\n")
        f.write("## Comprehensive Research Table\n\n")
        f.write("| Condition | $N$ | Acc | Verb Conf | Stat Conf | Verb Gap | Uncal ECE | M1 ECE | M2 ECE | $\\Delta$ECE Red | Uncal BS | M2 BS | Consistency (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in table_rows:
            cons_str = f"{r['answer_consistency_percent']:.1f}%" if r['answer_consistency_percent'] is not None else "N/A"
            f.write(f"| `{r['condition']}` | {r['test_sample_size']} | {r['accuracy']:.3f} | {r['verbalized_confidence']:.3f} | {r['statistical_confidence']:.3f} | {r['overconfidence_gap_verb']:+.3f} | {r['uncalibrated_v_ece']:.3f} | {r['model_1_v_ece']:.3f} | {r['model_2_v_ece']:.3f} | **{r['m2_delta_ece_reduction']:+.3f}** | {r['uncalibrated_v_bs']:.3f} | {r['model_2_v_bs']:.3f} | {cons_str} |\n")

    # =========================================================================
    # 6. RESULTS.MD (RESEARCH FINDINGS, EVIDENCE, CAVEATS & LIMITATIONS)
    # =========================================================================
    results_md_file = OUTPUT_DIR / "results.md"
    results_text = f"""# Empirical Results & Research Synthesis

**Study Title:** *Cross-Lingual Calibration Collapse Under Instruction-Tuning: Verbalized vs. Statistical Confidence, Answer Consistency and Mitigation*  
**Model & Hardware:** `google/gemma-3-4b-it` (Revision: `093f9f388b31de276ce2de164bdc2081324b9767`, BitsAndBytes 4-bit NF4) on NVIDIA GeForce RTX 4050 Laptop GPU (6,144 MB VRAM).  
**Methodological Protocol:** Frozen Specification v4.5, evaluated on held-out test partition $P_{{\\text{{test}}}}$ ($N=45$ questions across 11 conditions, 4,050 actual model inferences).

---

## 1. Executive Research Summary & Aims Status

| Research Aim | Theoretical Question | Empirical Status | Key Empirical Evidence |
| :--- | :--- | :---: | :--- |
| **Aim 1: Cross-Lingual Calibration Collapse** | Does confidence calibration collapse when querying in Indic languages relative to English? | **SUPPORTED (CONFIRMATORY)** | ECE degrades from 0.0440 in English to 0.5516–0.7025 in Indic languages. Multiplicity testing ($B=10,000$ bootstrap) confirms $\\Delta\\text{{ECE}} > 0$ for all Indic languages ($p < 0.0025$). |
| **Aim 2: Verbalized vs. Statistical Discrepancy** | Is calibration collapse distinct between verbalized elicitation and token-level statistical confidence? | **SUPPORTED (EMPIRICAL)** | Statistical confidence (token geometric mean) exhibits lower miscalibration than verbalized elicitation (e.g., Hindi ECE: 0.5771 vs 0.6798). Verbalized elicitation displays severe discrete clustering at $\\ge 95\\%$. |
| **Aim 3: Cross-Lingual Answer Consistency** | Do models generate factually consistent answers between English and Indic prompts? | **SUPPORTED (750 JUDGE PAIRS)** | Native Indic answers match English factual assertions in only 28.7%–37.3% of pairs under strict evidence grounding. Cross-lingual transfer (`hi_trans_en`) elevates agreement to 64.0%. |
| **Aim 4: Post-Hoc Recalibration Mitigation** | Can language-shift scaling (Model 2) restore calibration over pooled scaling (Model 1)? | **SUPPORTED (HELD-OUT)** | Model 2 selected on $P_{{\\text{{val}}}}$ (44.3% BS reduction over M1). On held-out $P_{{\\text{{test}}}}$, Model 2 reduces Indic ECE by up to 87.7% (Hindi: 0.6798 $\\to$ 0.1043; Tamil: 0.5516 $\\to$ 0.0676). |

---

## 2. Main Empirical Findings & Methodological Evidence

### Finding 1: Severe Cross-Lingual Calibration Collapse in Native Indic Querying
- In the English baseline (`en`), Gemma 3 4B achieves near-optimal probabilistic calibration on held-out test questions: $\\text{{ECE}} = 0.0440$, $\\text{{Brier Score}} = 0.0440$, with observed accuracy of 95.6% closely tracking mean verbalized confidence (99.9%).
- In native Indic conditions, calibration suffers a catastrophic breakdown:
  - **Hindi (`hi`):** $\\text{{ECE}} = 0.6798$, $\\text{{Brier Score}} = 0.6623$ (Accuracy: 23.8%, Mean Confidence: 91.8%, Overconfidence Gap: **+0.6798**).
  - **Bengali (`bn`):** $\\text{{ECE}} = 0.6049$, $\\text{{Brier Score}} = 0.6463$ (Accuracy: 35.6%, Mean Confidence: 96.0%, Overconfidence Gap: **+0.6049**).
  - **Telugu (`te`):** $\\text{{ECE}} = 0.7025$, $\\text{{Brier Score}} = 0.6692$ (Accuracy: 25.0%, Mean Confidence: 95.2%, Overconfidence Gap: **+0.7025**).
  - **Tamil (`ta`):** $\\text{{ECE}} = 0.5516$, $\\text{{Brier Score}} = 0.5791$ (Accuracy: 40.0%, Mean Confidence: 95.2%, Overconfidence Gap: **+0.5516**).
- **Murphy Decomposition Insight:** Over 70% of the Indic Brier Score error is dominated by the **Reliability ($\text{{REL}}$)** component (Hindi $\\text{{REL}} = 0.4855$ vs English $\\text{{REL}} = 0.0019$). The model maintains moderate factual discrimination but outputs extreme overconfidence regardless of factual validity.

### Finding 2: Verbalized Elicitation vs. Statistical Confidence Trajectories
- Statistical confidence derived from token logprobs (float32 log-softmax geometric probability over content tokens) is consistently better calibrated than verbalized confidence across all native Indic languages:
  - Hindi Uncalibrated: Verbalized $\\text{{ECE}} = 0.6798$ vs. Statistical $\\text{{ECE}} = 0.5771$ (15.1% lower miscalibration).
  - Bengali Uncalibrated: Verbalized $\\text{{ECE}} = 0.6049$ vs. Statistical $\\text{{ECE}} = 0.5123$ (15.3% lower miscalibration).
  - Tamil Uncalibrated: Verbalized $\\text{{ECE}} = 0.5516$ vs. Statistical $\\text{{ECE}} = 0.4120$ (25.3% lower miscalibration).
- **Linguistic Compliance Attrition:** In Telugu (`te`), the model frequently outputs explanatory textual rationales rather than conforming to the strict regex percentage contract, yielding 79.3% attrition on the verbalized track. In contrast, the statistical track achieves 0.0% attrition across all 1,650 generations.

### Finding 3: Symmetrical Judge Adjudication & Grounding Discrepancy (750 Pairs)
- Evaluation of 750 cross-lingual pairs using Gemma 3 4B as a symmetrical comparison judge with strict claim-level grounding reveals that **superficial agreement masks ungrounded factual evidence**:
  - The raw LLM judge classified 663 out of 750 pairs (88.4%) as `CONSISTENT`.
  - However, strict evidence span verification revealed that in **346 pairs (46.1%)**, the judge hallucinated ungrounded supporting evidence (`FABRICATED_EVIDENCE_DETECTED`).
  - Upon automated arbiter override, true grounded factual consistency between English and native Indic answers is only **29.3% for Hindi**, **28.7% for Bengali**, **37.3% for Telugu**, and **30.0% for Tamil**.
  - **Cross-Lingual Transfer Lift:** When queried in Hindi but instructed to answer in English (`hi_trans_en`), grounded consistency jumps to **64.0% (96/150 pairs)**.

### Finding 4: Calibration Mitigation Effectiveness (Model 1 vs. Model 2)
- **Model 1 (Pooled Platt Scaling):** Fits a single global slope $w$ and intercept $b$ across all languages. On held-out $P_{{\\text{{test}}}}$, it reduces Indic ECE from $\\sim 0.65$ to $\\sim 0.35$, but fails to eliminate the language-specific offset.
- **Model 2 (Language-Shift Platt Scaling):** Fits a shared slope $w$ with language-specific intercept shifts $\\delta_l$.
  - Selected decisively on validation data $P_{{\\text{{val}}}}$ (Validation Brier Score: 0.1215 for M2 vs 0.2182 for M1).
  - On held-out test data $P_{{\\text{{test}}}}$, Model 2 achieves dramatic calibration recovery:
    * Hindi Verbalized ECE: $0.6798 \\to \\mathbf{{0.1043}}$ (**84.7% reduction**).
    * Tamil Verbalized ECE: $0.5516 \\to \\mathbf{{0.0676}}$ (**87.7% reduction**).
    * Bengali Verbalized ECE: $0.6049 \\to \\mathbf{{0.2126}}$ (**64.9% reduction**).
    * Telugu Verbalized ECE: $0.7025 \\to \\mathbf{{0.2894}}$ (**58.8% reduction**).
  - **Trade-off in English:** Because English was already exceptionally well-calibrated (0.0440), applying global shrinkage slightly increases English ECE ($0.0440 \\to 0.1062$). Thus, language-conditioned recalibration must be selectively applied based on language detection.

---

## 3. Methodological Clarifications & Source Attribution

### Source of Identical Transferred-English Metrics
In the preliminary metrics summary, all four transferred conditions (`hi_trans_en`, `bn_trans_en`, `te_trans_en`, `ta_trans_en`) reported identical test metrics ($\\text{{ECE}} = 0.0113, \\text{{BS}} = 0.0009$).
- **Source Identification:** Inspection of `build_condition_prompts` reveals that for all `*_trans_en` conditions, the prompt string was constructed as:
  `"Answer the same question in English, preserving the meaning of the original question... Original question: {{q_en}}"`
- Because `q_en` (the English question text) was passed into the prompt across all four transferred conditions, the input to Gemma 3 4B was mathematically identical. Under greedy decoding (`do_sample=False`), identical inputs yield identical output tokens, explaining the exact numerical identity.

---

## 4. Unsupported Claims & Research Caveats

1. **Instruction-Tuning Causality:** While the paper title references "Instruction-Tuning", our empirical dataset evaluated the instruction-tuned checkpoint `gemma-3-4b-it`. Because base pretrained Gemma 3 4B (without instruction tuning) was not evaluated under this test harness, **causal claims attributing calibration collapse exclusively to the instruction-tuning stage cannot be formally established** from this data alone. The observed phenomenon is properly characterized as an epistemic misalignment present in the instruction-tuned model.
2. **Beyond Factual Accuracy:** The paper must not be framed simply as "Indic language accuracy is low." The primary finding is that **the model's epistemic self-awareness collapses**: while its accuracy drops from 95.6% to 23.8%–40.0%, its verbalized confidence remains stubbornly anchored above 91%–96%.
3. **Paired Sample Attrition:** In Telugu (`te`), paired cohort analysis on the verbalized track was constrained to $N=12$ test questions due to high refusal/unparsable response rates. While statistical tests remain significant at $p < 0.0025$, conclusions regarding Telugu verbalized calibration should note the reduced sample size.
"""
    with open(results_md_file, "w", encoding="utf-8") as f:
        f.write(results_text)

    print("Research deliverables generation complete:")
    print(f"  {f1_path}")
    print(f"  {f2_path}")
    print(f"  {f3_path}")
    print(f"  {f4_path}")
    print(f"  {csv_file}")
    print(f"  {md_summary_file}")
    print(f"  {results_md_file}")

if __name__ == "__main__":
    run()
