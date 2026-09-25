import os
import sys
import json
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

# Load the deep_dive_samples.json
with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\deep_dive_samples.json", "r", encoding="utf-8") as f:
    deep_dive = json.load(f)

sampled_gt = deep_dive["sampled_records"]["sampled_gt"]
s1_samples = deep_dive["sampled_records"]["s1_samples"]
s2_samples = deep_dive["sampled_records"]["s2_samples"]
s3_samples = deep_dive["sampled_records"]["s3_samples"]

print(f"Total sampled GT entries: {len(sampled_gt)}")
print(f"Available S1: {len(s1_samples)}, S2: {len(s2_samples)}, S3: {len(s3_samples)}")

# We also need more S1 records to inspect specific pairs
# Let's read a chunk of train_source1 to get the exact records for our sampled_gt S1 IDs
needed_s1_ids = {s1 for s1, _ in sampled_gt}
full_s1_dict = {}
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source1.tsv"), sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
        if eid in needed_s1_ids:
            full_s1_dict[eid] = {"entity_id": eid, "name": name, "address": addr, "country": ctry}

print(f"Retrieved {len(full_s1_dict)} matching S1 records.")

# Categories to populate:
# 1. Exact / near-exact name matches
# 2. Noisy name matches
# 3. Address variations
# 4. Missing address information
# 5. Multiple matches
# 6. Singleton Source 1 entities

examples = {
    "exact_near_exact_names": [],
    "noisy_name_matches": [],
    "address_variations": [],
    "missing_address": [],
    "multiple_matches": [],
    "singletons": []
}

for s1_id, match_ids in sampled_gt:
    s1 = full_s1_dict.get(s1_id)
    if not s1:
        continue
    
    if len(match_ids) == 0:
        if len(examples["singletons"]) < 8:
            examples["singletons"].append({
                "s1": s1,
                "matches": []
            })
        continue
        
    s1_name_clean = re.sub(r'[^a-zA-Z0-9]', '', s1["name"].lower())
    matched_recs = []
    for mid in match_ids:
        rec = s2_samples.get(mid) or s3_samples.get(mid)
        if rec:
            matched_recs.append({"entity_id": mid, **rec})
            
    if not matched_recs:
        continue
        
    # Check multiple matches
    if len(matched_recs) >= 3 and len(examples["multiple_matches"]) < 6:
        examples["multiple_matches"].append({
            "s1": s1,
            "match_count": len(matched_recs),
            "matches": matched_recs
        })
        
    # Check individual matches for properties
    for m in matched_recs:
        m_name_clean = re.sub(r'[^a-zA-Z0-9]', '', m["name"].lower())
        m_addr = m["address"]
        s1_addr = s1["address"]
        
        # Missing address
        if (not m_addr or m_addr.lower() == "nan") and len(examples["missing_address"]) < 6:
            examples["missing_address"].append({
                "s1": s1,
                "match": m,
                "note": "Source 2/3 address is missing/empty while Source 1 has address"
            })
            
        # Exact / near exact name
        if s1_name_clean == m_name_clean and len(examples["exact_near_exact_names"]) < 6:
            examples["exact_near_exact_names"].append({
                "s1": s1,
                "match": m,
                "note": "Exact or punctuation-only difference in business name"
            })
        elif s1_name_clean != m_name_clean:
            # Check if noisy name match
            # E.g. abbreviations, prefix additions, typos
            if len(examples["noisy_name_matches"]) < 8:
                examples["noisy_name_matches"].append({
                    "s1": s1,
                    "match": m,
                    "note": "Differences in abbreviations, prefixes, transliteration or spelling"
                })
                
        # Address variation
        if m_addr and m_addr.lower() != "nan" and s1_addr.lower() != m_addr.lower() and len(examples["address_variations"]) < 8:
            examples["address_variations"].append({
                "s1": s1,
                "match": m,
                "note": "Address formatted differently, abbreviated, or reordered"
            })

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\categorized_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print("Saved categorized_examples.json with counts:")
for cat, lst in examples.items():
    print(f"  {cat}: {len(lst)} examples")
