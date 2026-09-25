import os
import sys
import json
import nbformat as nbf

# Ensure notebooks directory exists
os.makedirs(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\notebooks", exist_ok=True)

# Load our precomputed stats
with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\stats_part1.json", "r", encoding="utf-8") as f:
    stats_part1 = json.load(f)

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\deep_dive_samples.json", "r", encoding="utf-8") as f:
    deep_dive = json.load(f)

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\categorized_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\pattern_analysis.json", "r", encoding="utf-8") as f:
    pattern_stats = json.load(f)

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\duplication_stats.json", "r", encoding="utf-8") as f:
    dup_stats = json.load(f)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.13"
}

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 1 — Comprehensive Data Inspection & Exploratory Data Analysis

### Executive Problem Overview
* **Objective:** Given business records from 3 independent sources (`Source 1`, `Source 2`, `Source 3`) with noisy and inconsistent fields, resolve which records across sources refer to the same real-world business entity.
* **Role of Source 1:** Deduplicated reference entity source. For every entity in Source 1, find all matching records in Source 2 and Source 3.
* **Cardinality:** A Source 1 entity can have **zero matches** (singleton), **one match**, or **multiple matches**.
* **Target Metric:** **Macro F0.5 Score** (Precision is weighted 2× over recall: $\\beta = 0.5$). False positives (wrong merges) are penalized much more heavily than missed matches.
* **Strict Challenge Rules:** Use **only** the supplied challenge dataset. No external APIs, databases, geocoding, internet searches, or data augmentation.
* **Open-ended Country Set:** Training data contains `US` and `India`. Test data introduces `France` (~15% of records). The pipeline must never be hard-coded to US/India.
"""))

# Imports & Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

# Set standard output encoding for multilingual text support if available
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve dataset directory path robustly
DATA_DIR = Path("dataset") if Path("dataset").exists() else Path("student_resource/dataset")
if not DATA_DIR.exists():
    DATA_DIR = Path("../student_resource/dataset")

print(f"Dataset root directory: {DATA_DIR.resolve()}")
assert DATA_DIR.exists(), f"Dataset directory not found at {DATA_DIR}"
"""))

# Section 1: File Inventory & Schema
cells.append(nbf.v4.new_markdown_cell("""---
## 1. File Inventory, Dataset Dimensions & Schema Validation

All files are tab-separated (`.tsv`). Reading requires explicit `sep="\\t"`.
Let us inspect the dimensions, column names, data types, missing values, and entity ID uniqueness across all 7 challenge files.
"""))

cells.append(nbf.v4.new_code_cell(f"""files_info = {json.dumps(stats_part1, indent=2)}

summary_rows = []
for key, info in files_info.items():
    summary_rows.append({{
        "File": info["file_name"],
        "Size (MB)": info["size_mb"],
        "Rows": f"{{info['total_rows']:,}}",
        "Columns": info["num_columns"],
        "Duplicate IDs": info["duplicate_entity_ids"],
        "Missing Names": info.get("null_counts", {{}}).get("business_name", 0),
        "Missing Addresses": info.get("null_counts", {{}}).get("business_address", 0),
    }})

summary_df = pd.DataFrame(summary_rows)
summary_df
"""))

cells.append(nbf.v4.new_markdown_cell("""### Key Findings from Dataset Dimensions:
1. **Total Records:** The dataset contains **24,233,194 records** (~12.5M train, ~11.7M test).
2. **Entity ID Uniqueness:** Duplicate entity IDs = **0** across all 7 files. Every entity ID is completely unique.
3. **Reference Source (Source 1):** Has **0 missing names** and **0 missing addresses** across both train and test.
4. **Source 2 & Source 3 Data Quality:**
   - Missing business names: Only 1 row in train_source2, 0 in all others.
   - Missing business addresses: **168,967 (3.36%)** in train_source2, **175,916 (3.33%)** in train_source3, **129,408 (2.65%)** in test_source2, and **136,098 (2.68%)** in test_source3.
"""))

# Section 2: Country Distribution
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Country Distribution & Open-Ended Test Set Analysis

The challenge statement highlights that country labels are open-ended.
Let us examine the exact country breakdown across all train and test files.
"""))

cells.append(nbf.v4.new_code_cell(f"""country_data = []
for key in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
    info = files_info[key]
    tot = info["total_rows"]
    counts = info["country_counts"]
    for ctry, cnt in counts.items():
        country_data.append({{
            "Split": "Train" if "train" in key else "Test",
            "Source File": info["file_name"],
            "Country": ctry,
            "Count": f"{{cnt:,}}",
            "Percentage": f"{{(cnt/tot)*100:.2f}}%"
        }})

pd.DataFrame(country_data)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Country Distribution Observations:
* **Training Set:** Exactly two countries: **US (~60%)** and **India (~40%)**.
* **Test Set:** Introduces **France (~14.5% - 15.0%)**, with **India (~47%)** and **US (~38%)**.
* **Crucial Implication:** Any country-specific logic (e.g. hardcoding 2-character US state abbreviations or Indian PIN regexes) will break or fail on France. All feature extractors and normalizers must be internationalized or gracefully handle open-ended country sets.
"""))

# Section 3: Business Name Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Business Name Analysis: Noise Patterns, Multilingual Scripts & Duplication

Let us examine the name length distribution, duplication rate in reference Source 1, punctuation artifacts, URL names, and multilingual scripts.
"""))

cells.append(nbf.v4.new_code_cell(f"""dup_info = {json.dumps(dup_stats, indent=2)}
pattern_info = {json.dumps(pattern_stats, indent=2)}

print(f"Total Source 1 Records: {{dup_info['total_s1']:,}}")
print(f"Unique Raw Names:       {{dup_info['unique_raw_names']:,}} ({{dup_info['duplicate_raw_name_rows']:,}} duplicate rows = {{dup_info['duplicate_raw_name_pct']}}%)")
print(f"Unique Normalized Names: {{dup_info['unique_norm_names']:,}} ({{dup_info['duplicate_norm_name_rows']:,}} duplicate rows = {{dup_info['duplicate_norm_name_pct']}}%)")
print("\\nTop 5 Most Frequent Business Names in Source 1:")
for name, cnt in dup_info["top_5_raw_names"]:
    print(f"  - {{name}}: {{cnt}} distinct entities")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Critical Name Duplication Insight:
Over **30.25%** of Source 1 records share their exact business name with at least one other distinct entity!
For example:
* `Primary Care Group`: 253 different entities
* `Ear Nose & Throat Group`: 251 different entities
* `Pediatric Group`: 222 different entities
* `Womens Health Group`: 220 different entities

> **Warning for F0.5 Metric:**
> If candidates or matches are selected purely based on business name similarity without strict address/location alignment, the pipeline will produce hundreds of false merges. Because F0.5 severely penalizes false positives, location-grounded matching is vital.
"""))

cells.append(nbf.v4.new_code_cell("""# Linguistic & Script Analysis
script_summary = []
for src, data in pattern_info.items():
    script_summary.append({
        "Dataset": src,
        "Leading Noise (--, <<, etc)": f"{data['leading_noise_symbols_count']:,}",
        "URL as Name (.com, .org)": f"{data['url_as_name_count']:,}",
        "Non-ASCII Names": f"{data['non_ascii_name_count']:,}",
        "Devanagari Script (Hindi)": f"{data['devanagari_script_count']:,}",
        "Tamil Script": f"{data['tamil_script_count']:,}",
        "Accented Latin (French)": f"{data['accented_latin_count']:,}",
    })
pd.DataFrame(script_summary)
"""))

# Section 4: Address Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Business Address Patterns & Variations

Let us analyze the structure of business addresses across sources and countries:
- Average length: ~52 characters in Source 1, ~48 in Sources 2 and 3.
- Address missingness: ~3.3% in Sources 2 & 3.
- Number presence: >90% of addresses contain numbers (house, plot, street, or postal code).
- Standard postal code presence: Only 5%–7% contain standalone 5- or 6-digit postal codes. The vast majority rely on street names, landmarks, localities, and cities/states.
"""))

cells.append(nbf.v4.new_code_cell("""# Address length statistics across all sources
addr_stats_rows = []
for key in ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]:
    info = files_info[key]
    astat = info["business_address_stats"]
    addr_stats_rows.append({
        "File": info["file_name"],
        "Missing Addresses": f"{astat['empty_count']:,} ({astat['empty_pct']}%)",
        "Min Length": astat["min_length"],
        "Max Length": astat["max_length"],
        "Mean Length": astat["mean_length"]
    })
pd.DataFrame(addr_stats_rows)
"""))

# Section 5: Ground Truth Deep Dive
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Ground Truth Matching Patterns & Match Cardinality

Let us examine the ground truth labels in `train_ground_truth.tsv` (2,206,821 Source 1 reference records):
* Singletons (0 matches)
* Single match (1 match)
* Multiple matches (>1 matches)
* Matches in Source 2 vs Source 3
* Cross-country consistency check
"""))

cells.append(nbf.v4.new_code_cell(f"""gt_stats = files_info["train_ground_truth"]
deep_dive_stats = {json.dumps(deep_dive, indent=2)}

print(f"Total Source 1 Reference Entities: {{gt_stats['total_rows']:,}}")
print(f"Total Matches to Source 2:        {{gt_stats['total_s2_matches']:,}}")
print(f"Total Matches to Source 3:        {{gt_stats['total_s3_matches']:,}}")
print(f"Total Matched Pairs:              {{gt_stats['total_s2_matches'] + gt_stats['total_s3_matches']:,}}")

print("\\nCardinality Distribution:")
print(f"  - Zero matches (Singletons):    {{gt_stats['singletons_zero_matches']:,}} ({{(gt_stats['singletons_zero_matches']/gt_stats['total_rows'])*100:.2f}}%)")
print(f"  - Exactly 1 match:              {{gt_stats['single_match_count']:,}} ({{(gt_stats['single_match_count']/gt_stats['total_rows'])*100:.2f}}%)")
print(f"  - Multiple matches (>1):        {{gt_stats['multiple_match_count']:,}} ({{(gt_stats['multiple_match_count']/gt_stats['total_rows'])*100:.2f}}%)")

print("\\nSource Combination Breakdown:")
src_break = deep_dive_stats["source_breakdown"]
print(f"  - S1 matching Neither (Singletons): {{src_break['singletons']:,}} ({{(src_break['singletons']/src_break['total'])*100:.2f}}%)")
print(f"  - S1 matching ONLY Source 2:       {{src_break['only_s2']:,}} ({{(src_break['only_s2']/src_break['total'])*100:.2f}}%)")
print(f"  - S1 matching ONLY Source 3:       {{src_break['only_s3']:,}} ({{(src_break['only_s3']/src_break['total'])*100:.2f}}%)")
print(f"  - S1 matching BOTH Source 2 & 3:   {{src_break['both_s2_and_s3']:,}} ({{(src_break['both_s2_and_s3']/src_break['total'])*100:.2f}}%)")

print("\\nCross-Country Consistency Check:")
ctry_check = deep_dive_stats["country_consistency_checked"]
print(f"  - Ground truth pairs checked:   {{ctry_check['total_pairs_checked']:,}}")
print(f"  - Exact country matches:         {{ctry_check['country_matches']:,}} (100.00%)")
print(f"  - Cross-country matches:        {{ctry_check['country_mismatches']}} (0.00%)")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Match Distribution Breakdown Table:
"""))

cells.append(nbf.v4.new_code_cell("""match_dist = gt_stats["gt_match_distribution"]
dist_rows = []
for k in sorted([int(x) for x in match_dist.keys()]):
    cnt = match_dist[str(k)]
    dist_rows.append({
        "Number of Matches": k,
        "S1 Entities": f"{cnt:,}",
        "Percentage": f"{(cnt / gt_stats['total_rows'])*100:.2f}%"
    })
pd.DataFrame(dist_rows)
"""))

# Section 6: Real Categorized Examples
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Real Dataset Examples by Category

Let us inspect verified ground-truth pairs representing all 6 required categories:
1. Exact / Near-exact Name Matches
2. Noisy Name Matches (abbreviations, typos, prefixes, URL names, transliterations)
3. Address Variations (reordered components, abbreviations, local script names)
4. Missing Address Information (Source 1 has address, Source 2/3 address is empty)
5. Multiple Matches (S1 matching 3+ distinct records across S2 and S3)
6. Singleton Source 1 Entities (no matching record in S2 or S3)
"""))

cells.append(nbf.v4.new_code_cell(f"""categorized_examples = {json.dumps(examples, indent=2)}

def display_examples(category_key, title):
    print("=" * 80)
    print(f"CATEGORY: {{title}}")
    print("=" * 80)
    for idx, item in enumerate(categorized_examples[category_key][:3], start=1):
        s1 = item.get("s1", {{}})
        print(f"\\n--- Example {{idx}} ---")
        print(f"Source 1: [{{s1.get('entity_id')}}] {{s1.get('name')}} | {{s1.get('address')}} | {{s1.get('country')}}")
        if "match" in item:
            m = item["match"]
            print(f"Match:    [{{m.get('entity_id')}}] {{m.get('name')}} | {{m.get('address')}} | {{m.get('country')}}")
            print(f"Note:     {{item.get('note')}}")
        elif "matches" in item:
            print(f"Total Matches: {{len(item['matches'])}}")
            for m in item["matches"][:3]:
                print(f"  -> [{{m.get('entity_id')}}] {{m.get('name')}} | {{m.get('address')}} | {{m.get('country')}}")

display_examples("exact_near_exact_names", "1. Exact / Near-Exact Name Matches")
display_examples("noisy_name_matches", "2. Noisy Name Matches")
display_examples("address_variations", "3. Address Variations")
display_examples("missing_address", "4. Missing Address Information")
display_examples("multiple_matches", "5. Multiple Matches")
display_examples("singletons", "6. Singleton Source 1 Entities")
"""))

# Section 7: Source Comparison & Quality Issues
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Comparative Source Analysis & Data Quality Issues

| Dimension | Source 1 (Reference) | Source 2 | Source 3 |
| :--- | :--- | :--- | :--- |
| **Role** | Deduplicated golden master | Uncurated source A | Uncurated source B |
| **Row Count (Train)** | 2,206,821 | 5,034,616 | 5,285,603 |
| **Row Count (Test)** | 1,732,544 | 4,887,273 | 5,082,316 |
| **Missing Addresses** | **0.00%** (0 nulls) | **3.36%** in train, **2.65%** in test | **3.33%** in train, **2.68%** in test |
| **Missing Names** | **0.00%** (0 nulls) | 1 record in train | 0 nulls |
| **Leading Noise Artifacts** | **0.00%** | **~0.91%** (`--`, `<<`, `>>`, `**`) | **~0.87%** (`--`, `<<`, `>>`, `**`) |
| **URL as Business Name** | **0.00%** | **~3.99%** (`.com`, `.org`, etc.) | **~3.96%** (`.com`, `.org`, etc.) |
| **Indic Scripts (in India)** | **0.00%** (English only) | **~9.2%** (Devanagari, Tamil, etc.) | **~5.3%** (Devanagari, Tamil, etc.) |
| **Address Reordering** | Standard Street -> City -> State | Frequent Inverted (State -> City -> Street) | Frequent Inverted & Landmark-based |

### Key Data-Quality Issues Identified:
1. **Transliteration Gap:** Source 1 names for India are purely in Latin script, whereas Sources 2 & 3 contain ~5%–9% Indic scripts (Devanagari, Tamil, Telugu). A character/token match without transliteration will fail on these pairs.
2. **URL Names:** Real businesses in S2/S3 often have website domains (e.g. `maurewilliamscolombier.com`) instead of formal business names.
3. **Punctuation & Prefix Garbage:** Systematic prefix noise (`-- Holloway Peak`, `<< Team Ecole`) is present in Sources 2 & 3.
4. **Missing Addresses:** ~3.3% of Source 2/3 records have no address at all. The matching pipeline must still resolve these using high-confidence name/URL matching.
5. **Address Permutation / Inversion:** Many addresses list state/city first (`CT, SLEEPY HOLLOW DRIVE, DANBURY`), requiring bag-of-words or token-set similarity rather than positional string matching.
"""))

# Section 8: Blueprint for Phase 2
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Strategic Blueprint for Phase 2

### 1. Preprocessing Requirements
- **Text Normalization:**
  - Strip leading/trailing noise characters (`--`, `<<`, `>>`, `**`, `//`, quotes).
  - Standardize whitespace and convert to lowercase.
  - Unicode normalization (NFKD) to strip accents (`é` -> `e`) for cross-lingual stability while retaining base characters.
  - Normalize URL business names: extract domain name tokens (e.g. `maurewilliamscolombier.com` -> `maure william colombier`).
  - Legal suffix mapping: Standardize (`pvt ltd`, `private limited`, `llc`, `l.l.c.`, `inc`, `corp`, `sarl`, `sas`).
- **Address Tokenization & Normalization:**
  - Expand street abbreviations (`rd` -> `road`, `st` -> `street`, `dr` -> `drive`, `blvd` -> `boulevard`, `ave` -> `avenue`, `r.` -> `rue`).
  - Extract numbers (house/plot/building numbers) and postal codes.
  - Sort address tokens or use set-based representations to handle address inversions.

### 2. Blocking & Candidate Generation Strategy
- **Rule 1: Strict Country Isolation (Lossless Block):**
  - Ground truth analysis verified that **100.0% of matches occur within the same country**.
  - Partitioning candidate generation strictly by country (`US`, `India`, `France`) reduces the comparison space immediately by ~60% with **zero recall loss**.
- **Rule 2: Multi-Pass Union Blocking:**
  - *Pass A (Standardized Name Tokens):* Top TF-IDF tokens or normalized first 2 significant words.
  - *Pass B (Cleaned Domain / Brand Core):* Stripping legal suffixes and matching core alphanumeric stem.
  - *Pass C (Address Location Token + Name Initial / N-gram):* Match numeric street/postal token + first letter of business name.
  - *Pass D (Indic Script / Phonetic Soundex / Metaphone):* For Indian entities, map phonetic representations or transliterated Latin tokens.
- **Recall Target:** Aim for >95% blocking recall while keeping candidates per S1 entity around 10–25 (manageable for downstream feature scoring).

### 3. Metric Alignment (Macro F0.5 Optimization)
- Because $\\beta = 0.5$, **Precision is weighted 2× over Recall**:
  $$F_{0.5} = \\frac{1.25 \\times \\text{Precision} \\times \\text{Recall}}{0.25 \\times \\text{Precision} + \\text{Recall}}$$
- A false positive (wrong merge) costs 4× more in the denominator balance than a false negative.
- Singletons represent **5.58%** of the dataset and earn a full **1.0** score when predicted empty, but drop to **0.0** if even a single false match is predicted. High classification thresholds are mandatory.
"""))

nb.cells = cells

# Save notebook
notebook_path = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\notebooks\01_data_inspection.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {notebook_path}")
