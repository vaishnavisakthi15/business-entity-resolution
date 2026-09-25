import os
import sys
import json
import re
from collections import Counter, defaultdict
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

print("--- Step 1: Mapping Entity IDs to Country and metadata ---")

# Let's map S1 entity_id -> country
s1_country = {}
s1_records_sample = {} # sample 100k S1 records for deep analysis
print("Loading train_source1 mapping...")
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source1.tsv"), sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
    for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
        s1_country[eid] = ctry
        if len(s1_records_sample) < 50000:
            s1_records_sample[eid] = {"name": name, "address": addr, "country": ctry}

print(f"Loaded {len(s1_country):,} S1 entity countries.")

# Let's inspect ground truth for S2 / S3 split and select candidates for example extraction
print("Analyzing train_ground_truth...")
s2_only_cnt = 0
s3_only_cnt = 0
both_s2_s3_cnt = 0
neither_cnt = 0

# Also track matched IDs we want to look up
lookup_s2_ids = set()
lookup_s3_ids = set()

# We'll sample 200 S1 entities with various match properties to inspect in full detail
sampled_gt = []

gt_chunk_size = 250000
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_ground_truth.tsv"), sep="\t", chunksize=gt_chunk_size, dtype=str, keep_default_na=False):
    for s1, m in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
        m = m.strip()
        if not m:
            neither_cnt += 1
            if len(sampled_gt) < 300 and neither_cnt % 5000 == 0:
                sampled_gt.append((s1, []))
        else:
            m_list = [x.strip() for x in m.split(",")]
            has_s2 = any(x.startswith("S2-") for x in m_list)
            has_s3 = any(x.startswith("S3-") for x in m_list)
            
            if has_s2 and has_s3:
                both_s2_s3_cnt += 1
            elif has_s2:
                s2_only_cnt += 1
            elif has_s3:
                s3_only_cnt += 1
                
            if len(sampled_gt) < 500:
                sampled_gt.append((s1, m_list))
                for mid in m_list:
                    if mid.startswith("S2-"):
                        lookup_s2_ids.add(mid)
                    elif mid.startswith("S3-"):
                        lookup_s3_ids.add(mid)

print(f"Ground Truth Source breakdown:")
print(f"  Neither (Singletons): {neither_cnt:,} ({neither_cnt / len(s1_country) * 100:.2f}%)")
print(f"  Only S2 matches:     {s2_only_cnt:,} ({s2_only_cnt / len(s1_country) * 100:.2f}%)")
print(f"  Only S3 matches:     {s3_only_cnt:,} ({s3_only_cnt / len(s1_country) * 100:.2f}%)")
print(f"  Both S2 & S3 matches:{both_s2_s3_cnt:,} ({both_s2_s3_cnt / len(s1_country) * 100:.2f}%)")

print(f"Selected {len(lookup_s2_ids)} S2 IDs and {len(lookup_s3_ids)} S3 IDs to fetch for deep example analysis.")

# Now fetch the records for these lookup IDs from train_source2 and train_source3
# Also check country alignment across matches
s2_records = {}
s2_country_mismatches = 0
s2_checked = 0

print("Scanning train_source2 for lookup IDs and cross-country check...")
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source2.tsv"), sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
        if eid in lookup_s2_ids:
            s2_records[eid] = {"name": name, "address": addr, "country": ctry}

print(f"Found {len(s2_records)} S2 records.")

s3_records = {}
print("Scanning train_source3 for lookup IDs...")
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source3.tsv"), sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
        if eid in lookup_s3_ids:
            s3_records[eid] = {"name": name, "address": addr, "country": ctry}

print(f"Found {len(s3_records)} S3 records.")

# Check cross-country matching on all retrieved pairs
country_mismatch_examples = []
total_pairs_checked = 0
perfect_country_matches = 0

for s1, m_list in sampled_gt:
    s1_ctry = s1_country.get(s1)
    for mid in m_list:
        mid_rec = s2_records.get(mid) or s3_records.get(mid)
        if mid_rec:
            total_pairs_checked += 1
            if mid_rec["country"] == s1_ctry:
                perfect_country_matches += 1
            else:
                country_mismatch_examples.append((s1, s1_ctry, mid, mid_rec["country"]))

print(f"Pairs checked for country match: {total_pairs_checked}")
print(f"Pairs with exact country match: {perfect_country_matches}")
print(f"Pairs with country mismatch: {len(country_mismatch_examples)}")

# Save detailed analysis
deep_dive = {
    "source_breakdown": {
        "singletons": neither_cnt,
        "only_s2": s2_only_cnt,
        "only_s3": s3_only_cnt,
        "both_s2_and_s3": both_s2_s3_cnt,
        "total": len(s1_country),
    },
    "country_consistency_checked": {
        "total_pairs_checked": total_pairs_checked,
        "country_matches": perfect_country_matches,
        "country_mismatches": len(country_mismatch_examples),
    },
    "sampled_records": {
        "sampled_gt": sampled_gt[:100],
        "s1_samples": {k: s1_records_sample[k] for k in list(s1_records_sample.keys())[:100]},
        "s2_samples": s2_records,
        "s3_samples": s3_records,
    }
}

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\deep_dive_samples.json", "w", encoding="utf-8") as f:
    json.dump(deep_dive, f, indent=2, ensure_ascii=False)

print("Saved deep_dive_samples.json")
