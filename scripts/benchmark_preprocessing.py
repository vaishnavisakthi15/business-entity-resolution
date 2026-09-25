import os
import sys
import time
import psutil
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import preprocess_tsv_to_parquet

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
OUTPUT_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\data\preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test 100,000 rows benchmark on Source 1
s1_tsv = os.path.join(BASE_DIR, "train", "train_source1.tsv")
s1_out = os.path.join(OUTPUT_DIR, "benchmark_train_source1_sample.parquet")

print("=== Benchmark 1: Preprocessing 100,000 rows of Source 1 ===")
res_s1 = preprocess_tsv_to_parquet(s1_tsv, s1_out, chunksize=50000, max_rows=100000)
print("Result S1:", res_s1)

# Test 100,000 rows benchmark on Source 2 (has missing addresses, Indic text, URLs, leading noise)
s2_tsv = os.path.join(BASE_DIR, "train", "train_source2.tsv")
s2_out = os.path.join(OUTPUT_DIR, "benchmark_train_source2_sample.parquet")

print("\n=== Benchmark 2: Preprocessing 100,000 rows of Source 2 ===")
res_s2 = preprocess_tsv_to_parquet(s2_tsv, s2_out, chunksize=50000, max_rows=100000)
print("Result S2:", res_s2)
