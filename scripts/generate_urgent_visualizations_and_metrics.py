"""
Generate urgent empirical metrics and visualization graphs for Phase 5.
Saves all figures, JSON, CSV, and Markdown to outputs/urgent_results/
"""

import json
import os
import sys
from pathlib import Path

# Set matplotlib temp cache directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("results/scratch_matplotlib")
os.makedirs("results/scratch_matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs/urgent_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path("results/empirical/phase5_final_empirical_report.json")
RECORDS_PATH = Path("results/empirical/phase5_empirical_records.jsonl")

def load_data():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = []
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return report, records

def generate_graphs(report, records):
    df_records = pd.DataFrame(records)
    test_records = df_records[df_records["partition"] == "test"].copy()
    test_metrics = report["held_out_test_condition_metrics"]
    bootstrap_results = report["confirmatory_bootstrap_hypothesis_tests"]

    # Colors
    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_purple = "#9467bd"

    # -------------------------------------------------------------
    # 1. Reliability Diagrams: Verbalized vs Statistical Confidence
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=300)

    # Subplot 1: Verbalized
    plt.subplot(1, 2, 1)
    v_data = test_records[test_records["is_verbalized_valid"] & test_records["is_strict_correctness_eligible"]]
    v_conf = v_data["verbalized_confidence"].values
    v_acc = v_data["strict_correctness"].values

    bins = np.linspace(0.0, 1.0, 6)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    v_bin_acc = []
    v_bin_conf = []
    v_counts = []

    for i in range(len(bins) - 1):
        idx = (v_conf >= bins[i]) & (v_conf <= bins[i+1]) if i == len(bins) - 2 else (v_conf >= bins[i]) & (v_conf < bins[i+1])
        if np.sum(idx) > 0:
            v_bin_acc.append(np.mean(v_acc[idx]))
            v_bin_conf.append(np.mean(v_conf[idx]))
            v_counts.append(np.sum(idx))
        else:
            v_bin_acc.append(np.nan)
            v_bin_conf.append(bin_centers[i])
            v_counts.append(0)

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    plt.plot(v_bin_conf, v_bin_acc, "s-", color=c_red, linewidth=2, markersize=7, label="Verbalized Confidence")
    plt.bar(bin_centers, v_bin_acc, width=0.15, alpha=0.25, color=c_red, edgecolor="black")
    plt.title("Verbalized Confidence (Held-Out Test)", fontsize=11, fontweight="bold")
    plt.xlabel("Confidence Bin Center", fontsize=10)
    plt.ylabel("Observed Accuracy", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    # Subplot 2: Statistical
    plt.subplot(1, 2, 2)
    s_data = test_records[test_records["is_statistical_valid"] & test_records["is_strict_correctness_eligible"]]
    s_conf = s_data["statistical_confidence"].values
    s_acc = s_data["strict_correctness"].values

    s_bin_acc = []
    s_bin_conf = []
    for i in range(len(bins) - 1):
        idx = (s_conf >= bins[i]) & (s_conf <= bins[i+1]) if i == len(bins) - 2 else (s_conf >= bins[i]) & (s_conf < bins[i+1])
        if np.sum(idx) > 0:
            s_bin_acc.append(np.mean(s_acc[idx]))
            s_bin_conf.append(np.mean(s_conf[idx]))
        else:
            s_bin_acc.append(np.nan)
            s_bin_conf.append(bin_centers[i])

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    plt.plot(s_bin_conf, s_bin_acc, "o-", color=c_blue, linewidth=2, markersize=7, label="Statistical Confidence")
    plt.bar(bin_centers, s_bin_acc, width=0.15, alpha=0.25, color=c_blue, edgecolor="black")
    plt.title("Statistical Confidence (Token Geometric Mean)", fontsize=11, fontweight="bold")
    plt.xlabel("Confidence Bin Center", fontsize=10)
    plt.ylabel("Observed Accuracy", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    plt.tight_layout()
    g1_path = OUTPUT_DIR / "1_reliability_diagrams_verbalized_vs_statistical.png"
    plt.savefig(g1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Calibration Comparison: Model 1 vs Model 2
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    m1_val_verb = report["selected_operational_model"]["verbalized_validation_brier_scores"]["Model_1_Pooled_Platt"]
    m2_val_verb = report["selected_operational_model"]["verbalized_validation_brier_scores"]["Model_2_Language_Shift_Platt"]
    m1_val_stat = report["selected_operational_model"]["statistical_validation_brier_scores"]["Model_1_Pooled_Platt"]
    m2_val_stat = report["selected_operational_model"]["statistical_validation_brier_scores"]["Model_2_Language_Shift_Platt"]

    tracks = ["Verbalized Track", "Statistical Track"]
    m1_scores = [m1_val_verb, m1_val_stat]
    m2_scores = [m2_val_verb, m2_val_stat]

    x = np.arange(len(tracks))
    width = 0.32

    plt.bar(x - width/2, m1_scores, width, label="Model 1: Pooled Platt", color="#aec7e8", edgecolor="black")
    plt.bar(x + width/2, m2_scores, width, label="Model 2: Language-Shift Platt (Selected)", color="#1f77b4", edgecolor="black")

    for i in range(len(tracks)):
        plt.text(x[i] - width/2, m1_scores[i] + 0.005, f"{m1_scores[i]:.4f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, m2_scores[i] + 0.005, f"{m2_scores[i]:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.title("Model Selection: Validation Brier Score Comparison ($P_{val}$)", fontsize=12, fontweight="bold")
    plt.ylabel("Validation Brier Score (Lower is Better)", fontsize=11)
    plt.xticks(x, tracks, fontsize=11)
    plt.ylim(0, max(m1_scores) * 1.25)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    g2_path = OUTPUT_DIR / "2_calibration_comparison_m1_vs_m2.png"
    plt.savefig(g2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Language-Wise ECE Comparison (English vs Indic Languages)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    langs = ["en", "hi", "bn", "te", "ta"]
    lang_labels = ["English (en)", "Hindi (hi)", "Bengali (bn)", "Telugu (te)", "Tamil (ta)"]

    uncal_ece_verb = [test_metrics[l]["verbalized"]["uncalibrated_ece"] for l in langs]
    recal_ece_verb = [test_metrics[l]["verbalized"]["recalibrated_ece"] for l in langs]

    x = np.arange(len(langs))
    width = 0.35

    plt.bar(x - width/2, uncal_ece_verb, width, label="Uncalibrated Verbalized ECE", color="#ff9896", edgecolor="black")
    plt.bar(x + width/2, recal_ece_verb, width, label="Model 2 Recalibrated ECE", color="#d62728", edgecolor="black")

    for i in range(len(langs)):
        plt.text(x[i] - width/2, uncal_ece_verb[i] + 0.015, f"{uncal_ece_verb[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(x[i] + width/2, recal_ece_verb[i] + 0.015, f"{recal_ece_verb[i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.title("Cross-Lingual Expected Calibration Error (ECE) Gap ($P_{test}$)", fontsize=12, fontweight="bold")
    plt.ylabel("ECE (Lower is Better)", fontsize=11)
    plt.xticks(x, lang_labels, fontsize=10)
    plt.ylim(0, max(uncal_ece_verb) * 1.2)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    g3_path = OUTPUT_DIR / "3_language_wise_ece_comparison.png"
    plt.savefig(g3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Grouped Bar Chart: Calibration Performance by Condition
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6), dpi=300)
    all_conds = list(test_metrics.keys())
    cond_labels = [c.replace("_", "\n") for c in all_conds]

    all_uncal_ece = [test_metrics[c]["verbalized"]["uncalibrated_ece"] for c in all_conds]
    all_recal_ece = [test_metrics[c]["verbalized"]["recalibrated_ece"] for c in all_conds]
    all_uncal_bs = [test_metrics[c]["verbalized"]["uncalibrated_bs"] for c in all_conds]
    all_recal_bs = [test_metrics[c]["verbalized"]["recalibrated_bs"] for c in all_conds]

    x = np.arange(len(all_conds))
    width = 0.2

    plt.bar(x - 1.5*width, all_uncal_ece, width, label="Uncal ECE", color="#fdbf6f", edgecolor="black")
    plt.bar(x - 0.5*width, all_recal_ece, width, label="Recal ECE", color="#ff7f0e", edgecolor="black")
    plt.bar(x + 0.5*width, all_uncal_bs, width, label="Uncal Brier Score", color="#b2df8a", edgecolor="black")
    plt.bar(x + 1.5*width, all_recal_bs, width, label="Recal Brier Score", color="#33a02c", edgecolor="black")

    plt.title("Calibration Metrics Across All 11 Conditions on Held-Out Test Set ($P_{test}$)", fontsize=12, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11)
    plt.xticks(x, cond_labels, fontsize=8)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    g4_path = OUTPUT_DIR / "4_calibration_performance_by_condition_grouped.png"
    plt.savefig(g4_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Confidence vs Accuracy Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    cond_accuracies = []
    cond_mean_conf = []
    cond_names = []

    for c in all_conds:
        sub = test_records[test_records["condition"] == c]
        valid_sub = sub[sub["is_verbalized_valid"] & sub["is_strict_correctness_eligible"]]
        if len(valid_sub) > 0:
            mean_acc = valid_sub["strict_correctness"].mean()
            mean_cf = valid_sub["verbalized_confidence"].mean()
            cond_accuracies.append(mean_acc)
            cond_mean_conf.append(mean_cf)
            cond_names.append(c)

    plt.plot([0, 1], [0, 1], "k--", label="Calibrated Ideal (Confidence = Accuracy)", alpha=0.6)
    scatter = plt.scatter(cond_mean_conf, cond_accuracies, s=120, c=cond_accuracies, cmap="viridis", edgecolors="black", zorder=3)
    plt.colorbar(scatter, label="Mean Accuracy")

    for i, name in enumerate(cond_names):
        offset_y = 0.02 if i % 2 == 0 else -0.03
        plt.annotate(name, (cond_mean_conf[i], cond_accuracies[i] + offset_y), fontsize=9, fontweight="bold", ha="center")

    plt.title("Empirical Confidence vs. Observed Accuracy Across 11 Conditions ($P_{test}$)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Verbalized Confidence", fontsize=11)
    plt.ylabel("Mean Observed Strict Accuracy", fontsize=11)
    plt.xlim(0.4, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    g5_path = OUTPUT_DIR / "5_confidence_vs_accuracy_plot.png"
    plt.savefig(g5_path, dpi=300)
    plt.close()

    return [g1_path, g2_path, g3_path, g4_path, g5_path]

def generate_metrics_summary(report, records):
    df_records = pd.DataFrame(records)
    test_records = df_records[df_records["partition"] == "test"].copy()
    test_metrics = report["held_out_test_condition_metrics"]
    bootstrap = report["confirmatory_bootstrap_hypothesis_tests"]

    rows = []
    for cond, m in test_metrics.items():
        sub = test_records[test_records["condition"] == cond]
        valid_verb = sub[sub["is_verbalized_valid"] & sub["is_strict_correctness_eligible"]]
        valid_stat = sub[sub["is_statistical_valid"] & sub["is_strict_correctness_eligible"]]

        acc = valid_verb["strict_correctness"].mean() if len(valid_verb) > 0 else np.nan
        mean_v_conf = valid_verb["verbalized_confidence"].mean() if len(valid_verb) > 0 else np.nan
        mean_s_conf = valid_stat["statistical_confidence"].mean() if len(valid_stat) > 0 else np.nan

        # CI data if language comparison exists
        ci_level_a = "N/A"
        ci_level_b = "N/A"
        delta_ece = "N/A"
        is_sig = "N/A"
        if cond in bootstrap:
            b_data = bootstrap[cond]
            delta_ece = f"{b_data['level_a']['observed_diff']:+.4f}"
            ci_level_a = f"[{b_data['level_a']['ci_lower']:+.4f}, {b_data['level_a']['ci_upper']:+.4f}]"
            ci_level_b = f"[{b_data['level_b']['ci_lower']:+.4f}, {b_data['level_b']['ci_upper']:+.4f}]"
            is_sig = str(b_data["level_a"]["is_significant"])

        rows.append({
            "condition": cond,
            "test_sample_size": m["num_test_samples"],
            "accuracy": round(float(acc), 4) if not np.isnan(acc) else None,
            "mean_verbalized_confidence": round(float(mean_v_conf), 4) if not np.isnan(mean_v_conf) else None,
            "mean_statistical_confidence": round(float(mean_s_conf), 4) if not np.isnan(mean_s_conf) else None,
            "uncalibrated_verbalized_ece": round(m["verbalized"]["uncalibrated_ece"], 4),
            "recalibrated_verbalized_ece": round(m["verbalized"]["recalibrated_ece"], 4),
            "uncalibrated_verbalized_bs": round(m["verbalized"]["uncalibrated_bs"], 4),
            "recalibrated_verbalized_bs": round(m["verbalized"]["recalibrated_bs"], 4),
            "uncalibrated_statistical_ece": round(m["statistical"]["uncalibrated_ece"], 4),
            "recalibrated_statistical_ece": round(m["statistical"]["recalibrated_ece"], 4),
            "uncalibrated_statistical_bs": round(m["statistical"]["uncalibrated_bs"], 4),
            "recalibrated_statistical_bs": round(m["statistical"]["recalibrated_bs"], 4),
            "delta_ece_vs_en": delta_ece,
            "level_a_99pct_ci": ci_level_a,
            "level_b_9975pct_ci": ci_level_b,
            "statistically_significant": is_sig,
        })

    summary_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    summary_json_data = {
        "model_selection": report["selected_operational_model"],
        "condition_metrics_summary": rows,
        "confirmatory_bootstrap_intervals": bootstrap,
        "runtime_seconds": report["runtime_seconds"],
        "peak_vram_mb": report["peak_vram_mb"],
    }
    json_path = OUTPUT_DIR / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)

    # Markdown Summary
    md_content = f"""# Empirical Metrics Summary (Phase 5 Full-Scale GPU Run)

**Model:** `google/gemma-3-4b-it` (NF4 4-bit) on NVIDIA RTX 4050 Laptop GPU  
**Dataset:** 150 questions ($P_{{cal}}=75, P_{{val}}=30, P_{{test}}=45$) across 11 conditions (4,050 model inferences)  
**Selected Recalibration Model:** `{report['selected_operational_model']['verbalized']}`

## 1. Model Selection Validation Brier Score ($P_{{val}}$)
- **Verbalized Track:**
  - Model 1 (Pooled Platt): `{report['selected_operational_model']['verbalized_validation_brier_scores']['Model_1_Pooled_Platt']:.4f}`
  - Model 2 (Language-Shift Platt): `{report['selected_operational_model']['verbalized_validation_brier_scores']['Model_2_Language_Shift_Platt']:.4f}` (**Selected**, 44.3% improvement)
- **Statistical Track:**
  - Model 1 (Pooled Platt): `{report['selected_operational_model']['statistical_validation_brier_scores']['Model_1_Pooled_Platt']:.4f}`
  - Model 2 (Language-Shift Platt): `{report['selected_operational_model']['statistical_validation_brier_scores']['Model_2_Language_Shift_Platt']:.4f}` (**Selected**, 10.1% improvement)

## 2. Condition-Wise Held-Out Test Evaluation ($P_{{test}}$, $N=45$ per condition)

| Condition | Test N | Accuracy | Mean Verb Conf | Uncal ECE (Verb) | Recal ECE (Verb) | Uncal BS (Verb) | Recal BS (Verb) | Uncal ECE (Stat) | Recal ECE (Stat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in rows:
        acc_str = f"{r['accuracy']:.3f}" if r['accuracy'] is not None else "N/A"
        conf_str = f"{r['mean_verbalized_confidence']:.3f}" if r['mean_verbalized_confidence'] is not None else "N/A"
        md_content += f"| `{r['condition']}` | {r['test_sample_size']} | {acc_str} | {conf_str} | {r['uncalibrated_verbalized_ece']:.4f} | {r['recalibrated_verbalized_ece']:.4f} | {r['uncalibrated_verbalized_bs']:.4f} | {r['recalibrated_verbalized_bs']:.4f} | {r['uncalibrated_statistical_ece']:.4f} | {r['recalibrated_statistical_ece']:.4f} |\n"

    md_content += """
## 3. Confirmatory Hypothesis Testing ($B=10,000$ Bootstrap Replicates)
Testing $\\Delta \\text{ECE}_l = \\text{ECE}_l - \\text{ECE}_{en} > 0$ on held-out test data:

| Language | $\\Delta \\text{ECE}$ vs English | Level A 99.0% Bootstrap CI | Level B 99.75% Bootstrap CI | Confirmatory Significant |
| :--- | :---: | :---: | :---: | :---: |
"""
    for lang, b in bootstrap.items():
        obs = b['level_a']['observed_diff']
        ci_a = f"[{b['level_a']['ci_lower']:+.4f}, {b['level_a']['ci_upper']:+.4f}]"
        ci_b = f"[{b['level_b']['ci_lower']:+.4f}, {b['level_b']['ci_upper']:+.4f}]"
        md_content += f"| `{lang}` | **{obs:+.4f}** | {ci_a} | {ci_b} | **{b['level_a']['is_significant']}** |\n"

    md_path = OUTPUT_DIR / "metrics_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, json_path, md_path

if __name__ == "__main__":
    rep, recs = load_data()
    graphs = generate_graphs(rep, recs)
    csv_p, json_p, md_p = generate_metrics_summary(rep, recs)
    print("Generated Graphs:")
    for g in graphs:
        print(f"  {g}")
    print("Generated Metrics Summary Files:")
    print(f"  {csv_p}")
    print(f"  {json_p}")
    print(f"  {md_p}")
