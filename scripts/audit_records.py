import io
import json
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

with open("results/test_predictions.jsonl", "r", encoding="utf-8") as f:
    recs = [json.loads(line) for line in f if line.strip()]

print(f"Total records in results/test_predictions.jsonl: {len(recs)}\n")

for i in range(0, len(recs), 2):
    r_orig = recs[i]
    r_eng = recs[i+1]
    cd = r_orig["comparison_details"]
    
    print("=" * 80)
    print(f"LANGUAGE PAIR: {r_orig['detected_language'].upper()} ({r_orig['detected_language_name']}) vs EN")
    print("=" * 80)
    print(f"Original Answer ({r_orig['answer_language']}):")
    print(f"  {r_orig['answer_text']}")
    print(f"English Answer ({r_eng['answer_language']}):")
    print(f"  {r_eng['answer_text']}\n")
    
    print(f"Raw Judge Classification    : {cd['raw_judge_classification']}")
    print(f"Final Arbiter Classification: {cd['final_arbiter_classification']}")
    print(f"Decision Source             : {cd['classification_decision_source']}")
    print(f"Arbiter Override Reason     : {cd['arbiter_override_reason']}")
    print(f"Evidence Validation Status  : {cd['evidence_validation_status']}")
    print(f"Overall Rationale           : {cd['overall_rationale']}\n")
    
    print(f"Claims ({len(cd['claims'])} total):")
    for c in cd["claims"]:
        print(f"  - [{c['claim_id']}] Relationship: {c['relationship']}")
        print(f"    Interpreted Claim : {c['interpreted_claim']}")
        print(f"    Orig Evidence Span: {repr(c['original_evidence_span'])} (Valid={c['is_original_evidence_valid']})")
        print(f"    Eng Evidence Span : {repr(c['english_evidence_span'])} (Valid={c['is_english_evidence_valid']})")
        print(f"    Validation Error  : {c['validation_error']}")
        print(f"    Explanation       : {c['explanation']}")
    
    df = cd["deterministic_features"]
    print(f"\nDeterministic Discrepancy Candidates ({len(df['candidate_discrepancies'])} total):")
    if not df["candidate_discrepancies"]:
        print("  None")
    for d in df["candidate_discrepancies"]:
        print(f"  - Category: {d['category']}")
        print(f"    Orig Value: {repr(d['original_value'])}, Context: {repr(d['original_context'])}")
        print(f"    Eng Value : {repr(d['english_value'])}, Context: {repr(d['english_context'])}")
        print(f"    Accounted for by Comparator: {d.get('accounted_for_by_comparator', d.get('resolved_by_judge'))}")
        print(f"    Comparator Resolution Status: {d.get('comparator_resolution_status', d.get('resolution_status'))}")
        print(f"    External Verification Status: {d.get('external_verification_status', 'N/A')}")
        print(f"    Comparator Resolution Reason: {d.get('comparator_resolution_reason', d.get('resolution_reason'))}")
        print(f"    Resolving Claim ID          : {d.get('resolving_claim_id')}")
        print(f"    Judge Assessment            : {d.get('judge_assessment')}")
    print("\n")
