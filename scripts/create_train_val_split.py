import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
SPLITS_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\data\splits"
os.makedirs(SPLITS_DIR, exist_ok=True)

print("=== Step 1: Loading Source 1 entity IDs and country for splitting ===")
t0 = time.time()

# Read only entity_id and country from train_source1
s1_path = os.path.join(BASE_DIR, "train", "train_source1.tsv")
df_s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], dtype=str)
print(f"Loaded {len(df_s1):,} Source 1 records in {time.time() - t0:.2f}s.")

# Check country distribution
print("Source 1 Country breakdown:")
print(df_s1["country"].value_counts(normalize=True).round(4) * 100)

# Perform stratified 80/20 split on Source 1 entities with fixed random seed 42
print("\n=== Step 2: Creating reproducible 80/20 train/validation split (seed=42) ===")
train_ids, val_ids = train_test_split(
    df_s1["entity_id"].values,
    test_size=0.20,
    random_state=42,
    stratify=df_s1["country"].values
)

print(f"Total Source 1 entities: {len(df_s1):,}")
print(f"Training split:          {len(train_ids):,} ({len(train_ids) / len(df_s1) * 100:.2f}%)")
print(f"Validation split:        {len(val_ids):,} ({len(val_ids) / len(df_s1) * 100:.2f}%)")

# Construct split dataframe
split_df = pd.DataFrame({
    "source1_entity_id": df_s1["entity_id"],
    "country": df_s1["country"],
    "split": np.where(df_s1["entity_id"].isin(set(train_ids)), "train", "val")
})

# Verify split counts by country
print("\nSplit verification by country:")
print(pd.crosstab(split_df["split"], split_df["country"], margins=True))

# Save to parquet and tsv for fast, reproducible access
parquet_path = os.path.join(SPLITS_DIR, "train_val_split_s1.parquet")
tsv_path = os.path.join(SPLITS_DIR, "train_val_split_s1.tsv")

split_df.to_parquet(parquet_path, index=False, compression="snappy")
split_df.to_csv(tsv_path, sep="\t", index=False)

print(f"\nSaved split assignments:")
print(f"  Parquet: {parquet_path} ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")
print(f"  TSV:     {tsv_path} ({os.path.getsize(tsv_path) / (1024*1024):.2f} MB)")

# Verify reload
check_df = pd.read_parquet(parquet_path)
assert len(check_df) == len(df_s1)
assert set(check_df["split"].unique()) == {"train", "val"}
print("\nValidation split verified successfully and ready for Phase 2 pipeline.")
