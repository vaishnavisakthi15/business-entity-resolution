import os
import time

base_dir = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
files = [
    os.path.join(base_dir, "train", "train_source1.tsv"),
    os.path.join(base_dir, "train", "train_source2.tsv"),
    os.path.join(base_dir, "train", "train_source3.tsv"),
    os.path.join(base_dir, "train", "train_ground_truth.tsv"),
    os.path.join(base_dir, "test", "test_source1.tsv"),
    os.path.join(base_dir, "test", "test_source2.tsv"),
    os.path.join(base_dir, "test", "test_source3.tsv"),
]

for p in files:
    t0 = time.time()
    with open(p, "rb") as f:
        lines = 0
        buf_size = 1024 * 1024
        buf = f.read(buf_size)
        while buf:
            lines += buf.count(b"\n")
            buf = f.read(buf_size)
    dt = time.time() - t0
    print(f"{os.path.basename(p):25s}: {lines:>10,d} lines in {dt:.2f}s")
