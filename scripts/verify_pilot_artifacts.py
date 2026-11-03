import json

with open("results/pilot/pilot_diagnostic_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

table_a = rep["table_a_attrition"]
print("Diagnostic report Table A entries:")
total_gen = sum(r["total_generated"] for r in table_a)
strict_ok = sum(r["strict_correctness_eligible"] for r in table_a)
verb_ok = sum(r["verbalized_valid"] for r in table_a)
stat_ok = sum(r["statistical_valid"] for r in table_a)
dual_ok = sum(r["dual_track_eligible"] for r in table_a)

for r in table_a:
    print(f"{r['condition']}: gen={r['total_generated']}, strict={r['strict_correctness_eligible']}, verb={r['verbalized_valid']}, stat={r['statistical_valid']}, dual={r['dual_track_eligible']}")

print(f"SUMS in report table_a: gen={total_gen}, strict={strict_ok}, verb={verb_ok}, stat={stat_ok}, dual={dual_ok}")

with open("results/pilot/pilot_records.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total records in jsonl: {len(records)}")
strict_jsonl = sum(1 for r in records if r.get("is_strict_correctness_eligible"))
verb_jsonl = sum(1 for r in records if r.get("is_verbalized_valid"))
stat_jsonl = sum(1 for r in records if r.get("is_statistical_valid"))
dual_jsonl = sum(1 for r in records if r.get("is_strict_correctness_eligible") and r.get("is_verbalized_valid") and r.get("is_statistical_valid"))

print(f"DIRECT JSONL COUNTS: strict={strict_jsonl}, verb={verb_jsonl}, stat={stat_jsonl}, dual={dual_jsonl}")

claims_count = sum(len(r.get("claims", [])) for r in records)
print(f"Total claims in jsonl: {claims_count}")
print(f"Average claims per answer: {claims_count / len(records):.4f}")

with open("data/pilot_30_questions.jsonl", "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

print(f"Total questions in questions file: {len(questions)}")
gt_claims_count = sum(len(q.get("ground_truth_claims", [])) for q in questions)
print(f"Total ground truth claims in 30 questions: {gt_claims_count}")
print(f"Claims per question: {gt_claims_count / len(questions):.4f}")
print(f"Expected claims across 11 conditions: {gt_claims_count * 11}")
