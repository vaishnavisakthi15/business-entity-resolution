import os
import sys
import json
import time
from collections import Counter
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

FILES = {
    "train_source1": os.path.join(BASE_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(BASE_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(BASE_DIR, "train", "train_source3.tsv"),
    "train_ground_truth": os.path.join(BASE_DIR, "train", "train_ground_truth.tsv"),
    "test_source1": os.path.join(BASE_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(BASE_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(BASE_DIR, "test", "test_source3.tsv"),
}

results = {}

print("=== PART 1: FILE INSPECTION & BASIC STATS ===")

for key, path in FILES.items():
    t0 = time.time()
    size_bytes = os.path.getsize(path)
    print(f"\nProcessing {key} ({size_bytes / (1024*1024):.2f} MB)...")
    
    total_rows = 0
    null_counts = Counter()
    country_counts = Counter()
    
    name_len_sum = 0
    name_len_sq_sum = 0
    name_min_len = float('inf')
    name_max_len = 0
    name_empty_cnt = 0
    
    addr_len_sum = 0
    addr_len_sq_sum = 0
    addr_min_len = float('inf')
    addr_max_len = 0
    addr_empty_cnt = 0
    
    # Track unique entity_ids using a set of hashes or sample if memory bound
    # To be exact without memory blowup, let's see if we can store IDs in a set.
    # 5M strings of 12 chars is ~300MB in python set, totally fine!
    entity_ids = set()
    has_dup_ids = False
    dup_count = 0
    
    # Ground truth specific
    gt_match_counts = Counter()
    s2_match_total = 0
    s3_match_total = 0
    
    cols = None
    dtypes = {}
    
    chunksize = 250000
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False):
        if cols is None:
            cols = list(chunk.columns)
            dtypes = {col: "object (string)" for col in cols}
        
        n_chunk = len(chunk)
        total_rows += n_chunk
        
        if key == "train_ground_truth":
            # Columns: source1_entity_id, matched_entity_ids
            for s1, m in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
                if s1 in entity_ids:
                    has_dup_ids = True
                    dup_count += 1
                entity_ids.add(s1)
                
                m = m.strip()
                if not m:
                    gt_match_counts[0] += 1
                else:
                    m_list = m.split(",")
                    c = len(m_list)
                    gt_match_counts[c] += 1
                    for mid in m_list:
                        mid = mid.strip()
                        if mid.startswith("S2-"):
                            s2_match_total += 1
                        elif mid.startswith("S3-"):
                            s3_match_total += 1
        else:
            # Columns: entity_id, business_name, business_address, country
            for eid, name, addr, ctry in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
                if eid in entity_ids:
                    has_dup_ids = True
                    dup_count += 1
                entity_ids.add(eid)
                
                country_counts[ctry] += 1
                
                # Check null/empty
                name_str = str(name).strip() if name is not None else ""
                if name_str == "" or name_str.lower() == "nan":
                    name_empty_cnt += 1
                    null_counts["business_name"] += 1
                else:
                    l = len(name_str)
                    name_len_sum += l
                    name_len_sq_sum += l * l
                    if l < name_min_len: name_min_len = l
                    if l > name_max_len: name_max_len = l
                    
                addr_str = str(addr).strip() if addr is not None else ""
                if addr_str == "" or addr_str.lower() == "nan":
                    addr_empty_cnt += 1
                    null_counts["business_address"] += 1
                else:
                    l = len(addr_str)
                    addr_len_sum += l
                    addr_len_sq_sum += l * l
                    if l < addr_min_len: addr_min_len = l
                    if l > addr_max_len: addr_max_len = l
                    
                ctry_str = str(ctry).strip() if ctry is not None else ""
                if ctry_str == "" or ctry_str.lower() == "nan":
                    null_counts["country"] += 1
                    
    dt = time.time() - t0
    print(f"Done {key} in {dt:.2f}s: {total_rows:,} rows, {len(entity_ids):,} unique IDs, dups: {dup_count}")
    
    file_info = {
        "file_name": os.path.basename(path),
        "path": path,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "total_rows": total_rows,
        "num_columns": len(cols),
        "columns": cols,
        "dtypes": dtypes,
        "duplicate_entity_ids": dup_count,
        "null_counts": dict(null_counts),
    }
    
    if key == "train_ground_truth":
        file_info["gt_match_distribution"] = dict(gt_match_counts)
        file_info["total_s2_matches"] = s2_match_total
        file_info["total_s3_matches"] = s3_match_total
        file_info["singletons_zero_matches"] = gt_match_counts[0]
        file_info["single_match_count"] = gt_match_counts[1]
        file_info["multiple_match_count"] = sum(v for k, v in gt_match_counts.items() if k > 1)
    else:
        file_info["country_counts"] = dict(country_counts)
        valid_name_cnt = total_rows - name_empty_cnt
        valid_addr_cnt = total_rows - addr_empty_cnt
        file_info["business_name_stats"] = {
            "empty_count": name_empty_cnt,
            "empty_pct": round((name_empty_cnt / total_rows) * 100, 4) if total_rows else 0,
            "min_length": name_min_len if name_min_len != float('inf') else 0,
            "max_length": name_max_len,
            "mean_length": round(name_len_sum / valid_name_cnt, 2) if valid_name_cnt else 0,
        }
        file_info["business_address_stats"] = {
            "empty_count": addr_empty_cnt,
            "empty_pct": round((addr_empty_cnt / total_rows) * 100, 4) if total_rows else 0,
            "min_length": addr_min_len if addr_min_len != float('inf') else 0,
            "max_length": addr_max_len,
            "mean_length": round(addr_len_sum / valid_addr_cnt, 2) if valid_addr_cnt else 0,
        }
        
    results[key] = file_info

# Save intermediate stats
os.makedirs(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output", exist_ok=True)
with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\stats_part1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== SUMMARY OF PART 1 SAVED ===")
