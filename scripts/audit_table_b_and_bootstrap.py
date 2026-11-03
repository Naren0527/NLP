import json

with open("results/full_scale/phase5_final_execution_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

tb = rep["table_b_test_attrition"]
total_part = sum(r["partition_size"] for r in tb)
total_max = sum(r["max_paired_cohort"] for r in tb)
total_verb = sum(r["actual_paired_verbalized"] for r in tb)
total_stat = sum(r["actual_paired_statistical"] for r in tb)

print("Condition counts in report:")
for r in tb:
    print(f"{r['condition']:12}: verb={r['actual_paired_verbalized']}, stat={r['actual_paired_statistical']}")

print(f"Total part: {total_part}, max: {total_max}")
print(f"Total verb: {total_verb}, stat: {total_stat}")
print(f"Stat attrition %: {100.0 * (1.0 - total_stat / total_max):.4f}%")
print(f"Verb attrition %: {100.0 * (1.0 - total_verb / total_max):.4f}%")

# Now compute directly from results/full_scale/phase5_full_records.jsonl
with open("results/full_scale/phase5_full_records.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

test_records = [r for r in records if r["partition"] == "test"]
test_qids = sorted(list(set(r["question_id"] for r in test_records)))
print(f"\nUnique test question IDs in records: {len(test_qids)}")

elig = {}
for r in test_records:
    c = r["condition"]
    q = r["question_id"]
    elig[(c, q)] = {
        "strict": bool(r.get("is_strict_correctness_eligible", False)),
        "verb": bool(r.get("is_verbalized_valid", False)),
        "stat": bool(r.get("is_statistical_valid", False)),
    }

conditions = sorted(list(set(r["condition"] for r in test_records)))
direct_stat_sum = 0
direct_verb_sum = 0

print("\nDirect computation from raw ledger:")
for c in conditions:
    n_verb = 0
    n_stat = 0
    for q in test_qids:
        en_s = elig[("en", q)]["strict"]
        en_v = elig[("en", q)]["verb"]
        en_st = elig[("en", q)]["stat"]
        
        c_s = elig[(c, q)]["strict"]
        c_v = elig[(c, q)]["verb"]
        c_st = elig[(c, q)]["stat"]
        
        if c == "en":
            if en_s and en_v:
                n_verb += 1
            if en_s and en_st:
                n_stat += 1
        else:
            if en_s and en_v and c_s and c_v:
                n_verb += 1
            if en_s and en_st and c_s and c_st:
                n_stat += 1
                
    direct_verb_sum += n_verb
    direct_stat_sum += n_stat
    print(f"{c:12}: verb={n_verb}, stat={n_stat}")

print(f"\nDirect totals: verb={direct_verb_sum}, stat={direct_stat_sum}")
print(f"Direct stat attrition %: {100.0 * (1.0 - direct_stat_sum / (45 * 11)):.4f}%")
