import json

with open("results/pilot/pilot_records.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"Total answer records: {len(records)}")

# 1. Atomic claims check
total_claims_in_ledger = 0
claim_ids_set = set()
claims_by_record = []
is_independent_objects = True

for r in records:
    c_list = r.get("claims", [])
    total_claims_in_ledger += len(c_list)
    claims_by_record.append(len(c_list))
    for c in c_list:
        cid = c.get("claim_id")
        if cid in claim_ids_set:
            is_independent_objects = False
        claim_ids_set.add(cid)

print(f"Total atomic claims counted across all 330 records: {total_claims_in_ledger}")
print(f"Unique claim IDs: {len(claim_ids_set)}")
print(f"Are claim IDs unique per answer record: {is_independent_objects}")

with open("data/pilot_30_questions.jsonl", "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

manifest_gt_claims = sum(len(q.get("ground_truth_claims", [])) for q in questions)
print(f"Total GT claims in manifest: {manifest_gt_claims}")

# 2. Table B Paired Cohort calculation using Frozen v4.5 formulas
all_qids = [q["question_id"] for q in questions]
conditions = sorted(list(set(r["condition"] for r in records)))

# Build eligibility dict
# (cond, qid) -> dict(strict, verb, stat)
elig = {}
for r in records:
    c = r["condition"]
    q = r["question_id"]
    elig[(c, q)] = {
        "strict": bool(r.get("is_strict_correctness_eligible", False)),
        "verb": bool(r.get("is_verbalized_valid", False)),
        "stat": bool(r.get("is_statistical_valid", False)),
    }

print("\n--- TABLE B AUDIT ---")
print("Condition | CohortSize | MaxPossible | PairedVerb | PairedStat | AttrVerb% | AttrStat%")
for c in conditions:
    paired_verb_qids = []
    paired_stat_qids = []
    for q in all_qids:
        en_strict = elig[("en", q)]["strict"]
        en_verb = elig[("en", q)]["verb"]
        en_stat = elig[("en", q)]["stat"]
        
        c_strict = elig[(c, q)]["strict"]
        c_verb = elig[(c, q)]["verb"]
        c_stat = elig[(c, q)]["stat"]
        
        # Frozen v4.5 formulas:
        # If c == 'en':
        #   N_paired_verb(en, P) = E_{en,strict} * E_{en,verb}
        #   N_paired_stat(en, P) = E_{en,strict} * E_{en,stat}
        # Else:
        #   N_paired_verb(c, P) = E_{en,strict} * E_{en,verb} * E_{c,strict} * E_{c,verb}
        #   N_paired_stat(c, P) = E_{en,strict} * E_{en,stat} * E_{c,strict} * E_{c,stat}
        if c == "en":
            if en_strict and en_verb:
                paired_verb_qids.append(q)
            if en_strict and en_stat:
                paired_stat_qids.append(q)
        else:
            if en_strict and en_verb and c_strict and c_verb:
                paired_verb_qids.append(q)
            if en_strict and en_stat and c_strict and c_stat:
                paired_stat_qids.append(q)
                
    n_verb = len(paired_verb_qids)
    n_stat = len(paired_stat_qids)
    part_size = len(all_qids)
    attr_verb = round(100.0 * (1.0 - n_verb / part_size), 2)
    attr_stat = round(100.0 * (1.0 - n_stat / part_size), 2)
    print(f"{c:12} | {part_size:10} | {part_size:11} | {n_verb:10} | {n_stat:10} | {attr_verb:8}% | {attr_stat:8}%")
