import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

singletons = []
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_ground_truth.tsv"), sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
    for s1, m in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
        if not m.strip():
            singletons.append(s1)
            if len(singletons) >= 10:
                break
    if len(singletons) >= 10:
        break

print(f"Sampled singleton IDs: {singletons}")

# Find their S1 records
singleton_recs = []
for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source1.tsv"), sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
    for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
        if eid in singletons:
            singleton_recs.append({"entity_id": eid, "name": name, "address": addr, "country": ctry, "matched_entity_ids": ""})

# Also load categorized_examples.json and append singletons
with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\categorized_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

examples["singletons"] = singleton_recs

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\categorized_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"Updated categorized_examples.json with {len(singleton_recs)} singletons.")
for s in singleton_recs[:5]:
    print(" ", s)
