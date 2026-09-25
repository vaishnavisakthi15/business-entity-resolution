import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
files = {
    "train_source1": os.path.join(base_dir, "train", "train_source1.tsv"),
    "train_source2": os.path.join(base_dir, "train", "train_source2.tsv"),
    "train_source3": os.path.join(base_dir, "train", "train_source3.tsv"),
    "train_ground_truth": os.path.join(base_dir, "train", "train_ground_truth.tsv"),
    "test_source1": os.path.join(base_dir, "test", "test_source1.tsv"),
    "test_source2": os.path.join(base_dir, "test", "test_source2.tsv"),
    "test_source3": os.path.join(base_dir, "test", "test_source3.tsv"),
}

for name, path in files.items():
    print("=" * 60)
    print(f"FILE: {name} ({os.path.basename(path)})")
    print(f"PATH: {path}")
    df_sample = pd.read_csv(path, sep="\t", nrows=5)
    print("COLUMNS:", df_sample.columns.tolist())
    print("DTYPES:\n", df_sample.dtypes)
    print("FIRST 3 ROWS:")
    for idx, row in df_sample.head(3).iterrows():
        print(dict(row))
