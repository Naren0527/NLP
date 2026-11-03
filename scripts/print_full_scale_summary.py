import json

with open("results/full_scale/phase5_final_execution_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

print("Status:", rep["status"])
print("Total questions:", rep["questions_total"])
print("Total records:", rep["total_records"])
print("Total claims:", rep["total_claims"])
print("QC sampled:", rep["qc_audit_sampled"])
print("Partitions:", rep["partitions"])
print("Selected Operational Model:", rep["selected_operational_model"])

print("\n--- HELD-OUT TEST CONDITION METRICS (P_test) ---")
test_m = rep["held_out_test_condition_metrics"]
for cond in sorted(test_m.keys()):
    m_v = test_m[cond]["verbalized"]
    m_s = test_m[cond]["statistical"]
    print(f"{cond:12} | N={test_m[cond]['num_test_samples']:2d} | Verb Uncal ECE: {m_v['uncalibrated_ece']:.4f} -> Recal ECE: {m_v['recalibrated_ece']:.4f} | Stat Uncal ECE: {m_s['uncalibrated_ece']:.4f} -> Recal ECE: {m_s['recalibrated_ece']:.4f}")

print("\n--- CONFIRMATORY BOOTSTRAP HYPOTHESIS TESTS (P_test vs English) ---")
boot = rep["confirmatory_bootstrap_hypothesis_tests"]
for lang, res in boot.items():
    la = res["level_a"]
    lb = res["level_b"]
    print(f"{lang:4} Level A (K=5,  alpha=0.0100): diff={la['observed_diff']:+.4f}, 99.0% CI=[{la['ci_lower']:+.4f}, {la['ci_upper']:+.4f}], Significant={la['is_significant']}")
    print(f"{lang:4} Level B (K=20, alpha=0.0025): diff={lb['observed_diff']:+.4f}, 99.75% CI=[{lb['ci_lower']:+.4f}, {lb['ci_upper']:+.4f}], Significant={lb['is_significant']}")


print("\n=== TABLE A FULL-DATASET ATTRITION (N=150) ===")
total_gen = sum(r["total_generated"] for r in rep["table_a_full_attrition"])
strict_ok = sum(r["strict_correctness_eligible"] for r in rep["table_a_full_attrition"])
verb_ok = sum(r["verbalized_valid"] for r in rep["table_a_full_attrition"])
stat_ok = sum(r["statistical_valid"] for r in rep["table_a_full_attrition"])
dual_ok = sum(r["dual_track_eligible"] for r in rep["table_a_full_attrition"])

for r in rep["table_a_full_attrition"]:
    print(f"{r['condition']:12} | Gen: {r['total_generated']:3d} | Strict: {r['strict_correctness_eligible']:3d} | Verb: {r['verbalized_valid']:3d} | Stat: {r['statistical_valid']:3d} | Dual: {r['dual_track_eligible']:3d} | Attr: {r['attrition_rate_percent']:5.2f}%")
print(f"PIPELINE TOTAL | Gen: {total_gen:3d} | Strict: {strict_ok:3d} | Verb: {verb_ok:3d} | Stat: {stat_ok:3d} | Dual: {dual_ok:3d} | Attr: {100*(1 - dual_ok/total_gen):5.2f}%")

print("\n=== TABLE B HELD-OUT TEST PAIRED ATTRITION (N=45) ===")
for r in rep["table_b_test_attrition"]:
    print(f"{r['condition']:12} | Part: {r['partition_size']:2d} | Max: {r['max_paired_cohort']:2d} | PairedVerb: {r['actual_paired_verbalized']:2d} | PairedStat: {r['actual_paired_statistical']:2d} | AttrVerb: {r['attrition_rate_percent_verb']:5.2f}% | AttrStat: {r['attrition_rate_percent_stat']:5.2f}%")
