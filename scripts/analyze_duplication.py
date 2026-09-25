import os
import sys
import json
import re
from collections import Counter
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

print("Analyzing name duplication and address duplication in train_source1...")
raw_names = Counter()
norm_names = Counter()

raw_addresses = Counter()
norm_addresses = Counter()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for chunk in pd.read_csv(os.path.join(BASE_DIR, "train", "train_source1.tsv"), sep="\t", chunksize=500000, dtype=str, keep_default_na=False):
    for name, addr in zip(chunk["business_name"], chunk["business_address"]):
        raw_names[name] += 1
        n_name = normalize_text(name)
        norm_names[n_name] += 1
        
        raw_addresses[addr] += 1
        n_addr = normalize_text(addr)
        norm_addresses[n_addr] += 1

total_s1 = sum(raw_names.values())
unique_raw_names = len(raw_names)
unique_norm_names = len(norm_names)

unique_raw_addrs = len(raw_addresses)
unique_norm_addrs = len(norm_addresses)

print(f"Total S1 records: {total_s1:,}")
print(f"Unique raw names: {unique_raw_names:,} ({total_s1 - unique_raw_names:,} duplicate rows, {((total_s1 - unique_raw_names)/total_s1)*100:.2f}%)")
print(f"Unique norm names: {unique_norm_names:,} ({total_s1 - unique_norm_names:,} duplicate rows, {((total_s1 - unique_norm_names)/total_s1)*100:.2f}%)")

print(f"Top 5 most frequent raw names in S1:")
for n, c in raw_names.most_common(5):
    print(f"  {n}: {c} times")

print(f"Unique raw addresses: {unique_raw_addrs:,} ({total_s1 - unique_raw_addrs:,} duplicate rows)")
print(f"Unique norm addresses: {unique_norm_addrs:,} ({total_s1 - unique_norm_addrs:,} duplicate rows)")

dup_stats = {
    "total_s1": total_s1,
    "unique_raw_names": unique_raw_names,
    "duplicate_raw_name_rows": total_s1 - unique_raw_names,
    "duplicate_raw_name_pct": round(((total_s1 - unique_raw_names) / total_s1) * 100, 2),
    "unique_norm_names": unique_norm_names,
    "duplicate_norm_name_rows": total_s1 - unique_norm_names,
    "duplicate_norm_name_pct": round(((total_s1 - unique_norm_names) / total_s1) * 100, 2),
    "top_5_raw_names": raw_names.most_common(5),
    "unique_raw_addresses": unique_raw_addrs,
    "unique_norm_addresses": unique_norm_addrs,
    "top_5_raw_addresses": raw_addresses.most_common(5)
}

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\duplication_stats.json", "w", encoding="utf-8") as f:
    json.dump(dup_stats, f, indent=2, ensure_ascii=False)

print("Saved duplication_stats.json")
