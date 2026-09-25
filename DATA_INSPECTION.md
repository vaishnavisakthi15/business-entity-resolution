# Amazon ML Challenge 2026: Business Entity Resolution
# Phase 1 — Comprehensive Data Inspection Report

**Date:** September 2026  
**Evaluation Metric:** Macro $F_{0.5}$ Score (Precision weighted 2× over Recall)  
**Notebook Companion:** [`notebooks/01_data_inspection.ipynb`](file:///c:/Users/Vaishnavi/Downloads/6ab10eb3b23ba_student_resource/notebooks/01_data_inspection.ipynb)  
**Status:** Phase 1 Complete (Data Inspection & Exploratory Data Analysis Only — No model training or final predictions generated)

---

## 1. File Inventory & Exact File Paths

All provided files are Tab-Separated Values (`.tsv`) located in the student resource directory. The raw dataset remains strictly untouched and unmodified.

| File Key | Exact Absolute File Path | File Size (Bytes) | File Size (MB) |
| :--- | :--- | :--- | :--- |
| `train_source1` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv` | 210,069,713 | 200.34 MB |
| `train_source2` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv` | 489,301,488 | 466.63 MB |
| `train_source3` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv` | 503,705,637 | 480.37 MB |
| `train_ground_truth` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv` | 127,015,583 | 121.13 MB |
| `test_source1` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source1.tsv` | 175,022,086 | 166.91 MB |
| `test_source2` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source2.tsv` | 509,456,422 | 485.86 MB |
| `test_source3` | `c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source3.tsv` | 506,002,772 | 482.56 MB |

> **Critical File Reading Rule:** All input files must be read using `pd.read_csv(path, sep="\t")`. Business addresses, entity names, and ID lists contain commas; loading without `sep="\t"` treats each row as a single string.

---

## 2. Dataset Dimensions, Schema & Entity ID Uniqueness

Every file was inspected in full without sampling to determine exact row counts, column types, missing values, and entity ID uniqueness.

### Summary Table

| File Name | Rows | Columns | Column Names | Data Types | Missing Names | Missing Addresses | Duplicate Entity IDs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | **2,206,821** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | **0** (0.0%) | **0** (0.0%) | **0** (100% unique) |
| `train_source2.tsv` | **5,034,616** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | 1 (<0.01%) | **168,967** (3.36%) | **0** (100% unique) |
| `train_source3.tsv` | **5,285,603** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | **0** (0.0%) | **175,916** (3.33%) | **0** (100% unique) |
| `train_ground_truth.tsv` | **2,206,821** | 2 | `source1_entity_id, matched_entity_ids` | String (`object`) | N/A | N/A | **0** (100% unique) |
| `test_source1.tsv` | **1,732,544** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | **0** (0.0%) | **0** (0.0%) | **0** (100% unique) |
| `test_source2.tsv` | **4,887,273** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | **0** (0.0%) | **129,408** (2.65%) | **0** (100% unique) |
| `test_source3.tsv` | **5,082,316** | 4 | `entity_id, business_name, business_address, country` | String (`object`) | **0** (0.0%) | **136,098** (2.68%) | **0** (100% unique) |

### Key Observations:
1. **Total Records in Dataset:**
   - Training Set: **12,527,040 records** across Sources 1, 2, 3.
   - Test Set: **11,702,133 records** across Sources 1, 2, 3.
   - Combined Total: **24,229,173 entity records** + 2,206,821 ground-truth rows.
2. **Pristine Reference (Source 1):**
   - Source 1 contains **zero missing business names** and **zero missing business addresses** in both train and test splits. It acts as a clean, deduplicated anchor entity repository.
3. **Missing Addresses in Secondary Sources:**
   - Sources 2 and 3 exhibit missing addresses in **2.65% to 3.36%** of records (~130k to 176k records per file). Business names in these records are almost 100% populated.
4. **Entity ID Integrity:**
   - Across all 24.2 million records, there are **0 duplicate entity IDs** within any file. ID prefixes strictly denote source origin (`S1-` for Source 1, `S2-` for Source 2, `S3-` for Source 3).

---

## 3. Country Distribution & Open-Ended Geography

A critical requirement of the challenge is not assuming a fixed country set. The actual dataset verifies this requirement:

| Split | File | Country | Record Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | `US` | 1,323,633 | 59.98% |
| **Train** | `train_source1.tsv` | `India` | 883,188 | 40.02% |
| **Train** | `train_source2.tsv` | `US` | 3,016,817 | 59.92% |
| **Train** | `train_source2.tsv` | `India` | 2,017,799 | 40.08% |
| **Train** | `train_source3.tsv` | `US` | 3,170,056 | 59.98% |
| **Train** | `train_source3.tsv` | `India` | 2,115,547 | 40.02% |
| **Test** | `test_source1.tsv` | `India` | 809,986 | **46.75%** |
| **Test** | `test_source1.tsv` | `US` | 663,106 | **38.27%** |
| **Test** | `test_source1.tsv` | `France` | 259,452 | **14.98%** |
| **Test** | `test_source2.tsv` | `India` | 2,312,565 | **47.32%** |
| **Test** | `test_source2.tsv` | `US` | 1,871,330 | **38.29%** |
| **Test** | `test_source2.tsv` | `France` | 703,378 | **14.39%** |
| **Test** | `test_source3.tsv` | `India` | 2,405,000 | **47.32%** |
| **Test** | `test_source3.tsv` | `US` | 1,945,701 | **38.28%** |
| **Test** | `test_source3.tsv` | `France` | 731,615 | **14.40%** |

### Crucial Findings on Country Distribution:
1. **The Test Set Shifts Significantly:**
   - While the training set is ~60% US and ~40% India, the test set shifts to **India as the largest cohort (47.3%)**, **US at 38.3%**, and **France introducing 1.7 million records (~14.5%–15.0%)**.
2. **Zero Cross-Country Matching in Ground Truth:**
   - Full validation across ground-truth pairs confirms that **100.00% of matches occur within the exact same country**. No US entity matches an Indian or French entity.
   - Consequently, **Country Partitioning is a 100% lossless blocking key**.
3. **Generalization Requirement:**
   - Hardcoding country-specific rules (such as US-only 2-letter state dictionaries or Indian 6-digit PIN regexes) will degrade performance on France. The pipeline must be generic or route by country label.

---

## 4. Business Name Analysis

### Length and Duplication Statistics

| Metric | Source 1 (Train) | Source 2 (Train) | Source 3 (Train) | Source 1 (Test) | Source 2 (Test) | Source 3 (Test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Character Length** | 24.03 | 25.10 | 25.20 | 23.84 | 25.70 | 25.66 |
| **Min Length** | 3 | 2 | 2 | 3 | 2 | 2 |
| **Max Length** | 105 | 104 | 123 | 92 | 102 | 103 |
| **Empty Names** | 0 (0.0%) | 1 (<0.001%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### The Name Duplication Phenomenon in Source 1
In `train_source1.tsv`:
- Total S1 records: **2,206,821**
- Unique raw names: **1,539,229** (667,592 duplicate rows = **30.25%**)
- Unique normalized names: **1,520,684** (686,137 duplicate rows = **31.09%**)

**Top Most Duplicated Names in Reference Source 1:**
- `Primary Care Group`: **253 distinct entities**
- `Ear Nose & Throat Group`: **251 distinct entities**
- `Pediatric Group`: **222 distinct entities**
- `Womens Health Group`: **220 distinct entities**
- `Physical Therapy Group`: **218 distinct entities**
- `Advanced Pain Care`: **184 distinct entities**

> **Direct Impact on $F_{0.5}$:**
> Because nearly 1 in 3 business names in the reference set is shared across different real-world businesses, **name similarity alone is insufficient**. Matching solely on name will merge hundreds of different clinics, shops, or agencies, creating disastrous false positives. High-precision address verification is mandatory.

### Linguistic, Noise and Script Patterns

Detailed pattern analysis across 100,000-record representative samples reveals:

1. **Systematic Leading Noise Prefix (~0.9% in S2 & S3):**
   - Source 1 contains 0 leading noise prefixes.
   - Sources 2 and 3 contain prefixes such as `--`, `<<`, `>>`, `**`, `//` (e.g., `-- Holloway Peak Inc Seafood`, `<< Team Ecole`).
2. **Website URL as Business Name (~4.0% in S2 & S3):**
   - Sources 2 and 3 frequently record website domains as the entity name (e.g. `maurewilliamscolombier.com`, `wilfordhancock.com`).
   - Source 1 stores the real company name (`Maure Williams Colombier Inc`).
3. **Multilingual Indic Scripts in Indian Records:**
   - Source 1 is **100% Latin-script transliterated** (English).
   - In Sources 2 & 3 (India), **~9.2% of names contain non-ASCII characters**:
     - Devanagari script (Hindi/Marathi): ~5.2% (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`, `मॉडर्न फाइनेंस`)
     - Tamil script: ~0.8% (e.g. `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`)
     - Telugu, Kannada, Gujarati, Malayalam, Bengali: ~3.2%
     - Mixed scripts: `Raj Investments எல்எல்பி`
4. **French Legal Entities & Diacritics in France Records:**
   - In test data for France: Accented Latin (`é`, `è`, `ê`, `à`, `ç`) appears in ~8% of names.
   - French legal suffixes dominate: `SARL` (4,224 / 100k), `SAS` (3,021 / 100k), `SCI`, `EURL`, `SA`.
5. **Common Legal Entity Suffixes Across Countries:**
   - US: `LLC` (16k/100k), `Inc` (11k/100k), `Corp` (1.6k/100k), `Corporation`, `Co`.
   - India: `Limited` (24k/100k), `Private` (20k/100k), `Pvt Ltd`, `LLP` (1.8k/100k).
   - France: `SARL`, `SAS`, `SCI`, `EURL`, `SASU`.

---

## 5. Business Address Analysis

### Length and Missingness Statistics

| Metric | Source 1 (Train) | Source 2 (Train) | Source 3 (Train) | Source 1 (Test) | Source 2 (Test) | Source 3 (Test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Character Length** | 52.07 | 47.83 | 48.32 | 57.21 | 51.78 | 50.08 |
| **Min Length** | 11 | 8 | 2 | 11 | 5 | 5 |
| **Max Length** | 256 | 249 | 240 | 268 | 269 | 267 |
| **Missing Addresses** | **0 (0.0%)** | **168,967 (3.36%)** | **175,916 (3.33%)** | **0 (0.0%)** | **129,408 (2.65%)** | **136,098 (2.68%)** |

### Address Components and Variations:
1. **High Presence of Numbers (>90%):**
   - 96.5% of Source 1 addresses and 90.7% of Source 2/3 addresses contain digits.
   - Digits correspond to building numbers, street numbers, plot/Khasra numbers (`KH NO. -570/13`), or postal codes.
2. **Postal Code Rarity (5% - 7%):**
   - Only 6.7% of Source 1 and 7.3% of Source 2/3 addresses contain explicit 5-digit or 6-digit standalone postal codes.
   - Therefore, **PIN/ZIP matching cannot be the sole blocking mechanism**; street tokens, city, and state are essential.
3. **Address Component Inversion:**
   - Source 1 follows standard hierarchical order: `Street -> City -> State/PIN`.
   - Sources 2 & 3 frequently invert this order, placing state or city first:
     - S1: `33 Sleepy Hollow Drive, Danbury, CT`
     - S2: `CT, SLEEPY HOLLOW DRIVE, DANBURY`
     - S1: `1064 Newton Rd, Unit 11, Iowa City, IA`
     - S2: `IA, Iowa City, 1064 Newton Rd, Unit 11`
4. **Address Transliteration into Regional Scripts:**
   - In Tamil Nadu records: Address state in S1 is `Tamil Nadu`, while S3 writes `தமிழ்நாடு` (Tamil script for Tamil Nadu).

---

## 6. Training Ground Truth Deep Dive

In `train_ground_truth.tsv` (2,206,821 Source 1 reference rows):

### Exact Ground-Truth Counts

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Source 1 Reference Entities** | **2,206,821** | 100.00% |
| **Entities with Zero Matches (Singletons)** | **123,247** | **5.58%** |
| **Entities with Exactly 1 Match** | **119,157** | **5.40%** |
| **Entities with Multiple Matches (>1 matches)** | **1,964,417** | **89.01%** |
| **Total Matches in Source 2** | **3,693,619** | — |
| **Total Matches in Source 3** | **3,944,746** | — |
| **Total Matched (S1, S2/S3) Pairs** | **7,638,365** | — |
| **Average Matches per Non-Singleton S1 Entity** | **3.66 matches** | — |
| **Maximum Matches for a Single S1 Entity** | **11 matches** | — |

### Full Match Cardinality Distribution

| Match Count per S1 Entity | Number of S1 Entities | Percentage |
| :---: | :---: | :---: |
| **0 (Singleton)** | **123,247** | **5.58%** |
| **1** | **119,157** | **5.40%** |
| **2** | **375,212** | **17.00%** |
| **3** | **530,841** | **24.05%** |
| **4** | **484,115** | **21.94%** |
| **5** | **321,957** | **14.59%** |
| **6** | **164,868** | **7.47%** |
| **7** | **63,968** | **2.90%** |
| **8** | **18,680** | **0.85%** |
| **9** | **4,205** | **0.19%** |
| **10** | **534** | **0.02%** |
| **11** | **37** | **<0.01%** |

### Breakdown of S1 Matching Behavior Across Sources

| Pattern | Count | Percentage of Reference S1 |
| :--- | :--- | :--- |
| **S1 matches BOTH Source 2 and Source 3** | **1,776,047** | **80.48%** |
| **S1 matches ONLY Source 3** | **164,498** | **7.45%** |
| **S1 matches ONLY Source 2** | **143,029** | **6.48%** |
| **S1 matches NEITHER (Singletons)** | **123,247** | **5.58%** |

---

## 7. Real Ground-Truth Examples Across 6 Categories

All examples below are extracted directly from the challenge files.

### Category 1: Exact / Near-Exact Name Matches
* **Example 1.1 (Case variation):**
  - **S1 (`S1-55344266`):** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu` | `India`
  - **S2 (`S2-197070651`):** `Raj Investments LLP` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu` | `India`
* **Example 1.2 (Punctuation and legal suffix variation):**
  - **S1 (`S1-546142636`):** `Crystal Staffing Solutions LLC` | `8706 Kentucky Derby Drive, Waxhaw, NC` | `US`
  - **S2 (`S2-582477216`):** `CRYSTAL STAFFING SOLUTIONS-L.L.C.` | `8706 KENTUCKY DERBY DR, WAXHAW, NC` | `US`
* **Example 1.3 (Typo in city name):**
  - **S1 (`S1-274126313`):** `Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA` | `US`
  - **S2 (`S2-736616474`):** `Obsidian,-LLC` | `3907 HAMILTON RD, DEER PARK CIYT, WA` | `US`

---

### Category 2: Noisy Name Matches (Typos, URLs, Transliteration, Prefix Noise)
* **Example 2.1 (Website URL domain as company name):**
  - **S1 (`S1-965667`):** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **S3 (`S3-11291185`):** `maurewilliamscolombier.com` | `Wayne Ave, Ticonderoga Townshiip, New York` | `US`
* **Example 2.2 (Typo in name):**
  - **S1 (`S1-965667`):** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **S2 (`S2-681193310`):** `Maure Wilblims Colombier Inc` (typo: `Wilblims`) | `[Empty Address]` | `US`
* **Example 2.3 (Transliteration into Tamil Script):**
  - **S1 (`S1-55344266`):** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu` | `India`
  - **S2 (`S2-249013014`):** `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu` | `India`
* **Example 2.4 (Mixed Script):**
  - **S1 (`S1-55344266`):** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu` | `India`
  - **S3 (`S3-478195123`):** `Raj Investments எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN` | `India`
* **Example 2.5 (Brand alias / Trade name variation):**
  - **S1 (`S1-965667`):** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **S3 (`S3-775321672`):** `Dréxkor` | `85 Wanye Avenue, Ticonderoga Townshiip, New York` | `US`

---

### Category 3: Address Variations (Reordering, Abbreviations, Missing Elements)
* **Example 3.1 (State/City first inverted format):**
  - **S1 (`S1-29845983`):** `Hendricks and Flowers Inc` | `33 Sleepy Hollow Drive, Danbury, CT` | `US`
  - **S2 (`S2-648035184`):** `Hendricks and  Flowers Inc` | `CT, SLEEPY HOLLOW DRIVE, DANBURY` | `US`
* **Example 3.2 (Street typo & Township variant):**
  - **S1 (`S1-965667`):** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **S3 (`S3-775321672`):** `Dréxkor` | `85 Wanye Avenue, Ticonderoga Townshiip, New York` | `US`
* **Example 3.3 (Regional script state name in address):**
  - **S1 (`S1-55344266`):** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu` | `India`
  - **S3 (`S3-384364074`):** `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு` | `India`

---

### Category 4: Missing Address Information
* **Example 4.1 (S2 address empty):**
  - **S1 (`S1-274126313`):** `Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA` | `US`
  - **S2 (`S2-680265918`):** `obsidian, llc` | `[EMPTY / NULL]` | `US`
* **Example 4.2 (S2 and S3 addresses empty for same S1):**
  - **S1 (`S1-965667`):** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **S2 (`S2-681193310`):** `Maure Wilblims Colombier Inc` | `[EMPTY / NULL]` | `US`
  - **S3 (`S3-860443364`):** `Maure Williams Inc Center` | `[EMPTY / NULL]` | `US`

---

### Category 5: Multiple Matches (1 S1 to Many S2/S3 Records)
* **Example 5.1 (`S1-965667` matches 5 records across S2 & S3):**
  - **S1:** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY` | `US`
  - **Match 1 (S2):** `S2-681193310` | `Maure Wilblims Colombier Inc` | `[EMPTY]` | `US`
  - **Match 2 (S2):** `S2-743505751` | `Maure Williams Colombier` | `[EMPTY]` | `US`
  - **Match 3 (S3):** `S3-775321672` | `Dréxkor` | `85 Wanye Avenue, Ticonderoga Townshiip, New York` | `US`
  - **Match 4 (S3):** `S3-11291185` | `maurewilliamscolombier.com` | `Wayne Ave, Ticonderoga Townshiip, New York` | `US`
  - **Match 5 (S3):** `S3-860443364` | `Maure Williams Inc Center` | `[EMPTY]` | `US`
* **Example 5.2 (`S1-55344266` matches 4 records across S2 & S3):**
  - **S1:** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu` | `India`
  - **Match 1 (S2):** `S2-249013014` | `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu` | `India`
  - **Match 2 (S2):** `S2-197070651` | `Raj Investments LLP` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu` | `India`
  - **Match 3 (S3):** `S3-478195123` | `Raj Investments எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN` | `India`
  - **Match 4 (S3):** `S3-384364074` | `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு` | `India`

---

### Category 6: Singleton Source 1 Entities (Zero True Matches)
* **Example 6.1 (India):**
  - **S1 (`S1-88224713`):** `Seaways & Associates` | `Plot No : 318 Village Baska, Taluka - Halol, Vadodara, Panch Mahals, Gujarat` | `India` | Matched IDs: `""`
* **Example 6.2 (US):**
  - **S1 (`S1-262997549`):** `Gabriella's Preferred Security` | `1013 Girard Avenue, Indianola, IA` | `US` | Matched IDs: `""`
* **Example 6.3 (US):**
  - **S1 (`S1-508022910`):** `Twyla's Liquor Corp` | `320 Flannery Lane, Silver Spring, MD` | `US` | Matched IDs: `""`
* **Example 6.4 (India):**
  - **S1 (`S1-830177070`):** `Namma Brand Pvt Ltd` | `Aarjees Buildingssasthancoil Road Thycaud, Trivandrum, Palakkad, Kerala` | `India` | Matched IDs: `""`
* **Example 6.5 (India):**
  - **S1 (`S1-666499407`):** `Vadapalani Projects Pvt. Ltd.` | `No.75, Natarajan Street, Dhanalakshmi Colony, Vadapalani, Chennai, Tamil Nadu` | `India` | Matched IDs: `""`

---

## 8. Comparative Characteristics: Source 1 vs Source 2 vs Source 3

| Feature | Source 1 (Reference) | Source 2 | Source 3 |
| :--- | :--- | :--- | :--- |
| **Dataset Role** | Golden reference (deduplicated) | Noisy multi-source feed 1 | Noisy multi-source feed 2 |
| **Row Count (Train)** | 2,206,821 | 5,034,616 | 5,285,603 |
| **Row Count (Test)** | 1,732,544 | 4,887,273 | 5,082,316 |
| **Missing Business Names** | 0 (0.00%) | 1 in train (<0.001%) | 0 (0.00%) |
| **Missing Business Addresses**| **0 (0.00%)** | **3.36% (train) / 2.65% (test)** | **3.33% (train) / 2.68% (test)** |
| **Punctuation Artifacts** | Clean punctuation | Systematic noise prefixes (`--`, `<<`) | Systematic noise prefixes (`--`, `<<`) |
| **URL Domain Names** | None (0.0%) | ~4.0% of records | ~4.0% of records |
| **Scripts Present (India)** | 100% Latin / English | ~9.2% non-ASCII (Devanagari, Tamil) | ~5.3% non-ASCII (Devanagari, Tamil) |
| **Address Structure** | Clean hierarchical order | Frequent inversion / abbreviations | Frequent inversion / landmarks |

---

## 9. Data Quality Issues Identified

1. **Indic Script Discrepancy:**
   - Source 1 contains only English (Latin) transliterations for Indian entities.
   - Sources 2 and 3 contain ~5%–9% regional Indian scripts (Devanagari, Tamil, Telugu). Standard character n-grams or edit distance will produce similarity = 0 without transliteration.
2. **Missing Addresses in Matching Targets:**
   - Over 168k addresses in S2 and 175k in S3 are null/empty. For these records, entity resolution must rely on high-confidence brand/name matching or domain names.
3. **URL as Business Name:**
   - ~4% of records in S2/S3 have names like `maurewilliamscolombier.com`. If not cleaned, `maurewilliamscolombier.com` has low token overlap with `Maure Williams Colombier Inc`.
4. **Extreme Name Duplication in Reference Set:**
   - 30.25% of Source 1 records share their exact name with other entities (e.g. hundreds of "Primary Care Group"). Blind name matching causes catastrophic precision loss.
5. **Inverted Address Formats:**
   - S2/S3 frequently reverse address elements (`State, Street, City` vs `Street, City, State`), rendering positional or sequential string metrics inaccurate unless set-based or token-sorted similarities are used.
6. **New Country in Test Set (`France`):**
   - 1.7 million test records belong to France, which is absent from training data. Any US/India-specific heuristics will fail on French addresses and legal entity designations (`SARL`, `SAS`).

---

## 10. Required Preprocessing Pipeline

Based strictly on the inspected data:

1. **Text Normalization:**
   - Strip leading punctuation artifacts: regex `^[-\<\>\*\#\@\_\~\+\/]+`.
   - Strip domain extensions: `.com`, `.org`, `.net`, `.in`, `.fr`, `.co`, `.io`.
   - Convert to lowercase and normalize consecutive whitespace.
   - Unicode NFKD normalization: decompose accents (`é` -> `e`, `à` -> `a`, `ô` -> `o`).
   - Standardize legal suffixes via canonical dictionary:
     - US: `inc, incorporated -> inc`; `llc, l.l.c. -> llc`; `corp, corporation -> corp`.
     - India: `pvt ltd, private limited, p ltd -> pvt ltd`; `llp -> llp`.
     - France: `sarl, s.a.r.l. -> sarl`; `sas, s.a.s. -> sas`; `sci -> sci`.
2. **Indic Script Transliteration:**
   - For records where `country == "India"`, apply phonetic romanization/transliteration (or script transliteration tables) to convert Devanagari and Tamil scripts into Latin characters.
3. **Address Normalization & Extraction:**
   - Expand standard road abbreviations: `rd -> road`, `st -> street`, `dr -> drive`, `ave -> avenue`, `blvd -> boulevard`, `ln -> lane`, `r. -> rue`.
   - Extract numeric address tokens (building/house numbers, plot numbers) into a dedicated feature set.
   - Create a token-sorted address representation to make matching invariant to component order.

---

## 11. Potential Blocking Strategies (Candidate Generation)

Given 1.73M test Source 1 entities and ~10M test Source 2/3 entities, comparing every S1 record against all S2/S3 records requires $1.73 \times 10^{13}$ pairwise comparisons—computationally impossible. A principled blocking strategy is required.

### Recommended Multi-Pass Union Blocking:
1. **Pass 0 — Strict Country Partitioning (Lossless):**
   - Candidate pairs are generated **strictly within the same country**.
   - Validated: 100% of ground-truth matches share the identical country.
   - Immediately eliminates ~60% of spurious cross-country pairs with zero recall loss.
2. **Pass 1 — Normalized Core Name Stem / Brand Prefix:**
   - Group records by their primary name token (after legal suffix removal and normalization) or first 2 significant tokens.
3. **Pass 2 — Cleaned Domain Stem:**
   - Extract the core alphanumeric domain stem (e.g. `maurewilliamscolombier`) and match against stripped business names.
4. **Pass 3 — Address Number + Name Initial (Locality Anchor):**
   - For common business names (e.g. "Primary Care Group"), block by: `(Address Number + State/City)` + `First 2 letters of Business Name`.
5. **Pass 4 — MinHash / TF-IDF Cosine Candidate Retrieval:**
   - Sparse character 3-gram or token TF-IDF indexed by country to capture spelling variants and typos with high recall.
6. **Candidate Union & Budgeting:**
   - Take the union of passes, keeping the top 10–25 candidates per S1 entity to feed into `candidate_pairs.tsv` and the downstream matching model.

---

## 12. Final Executive Summary

### A. Dataset Structure
- Total records: **24,229,173** across 7 `.tsv` files.
- Train: Source 1 (2,206,821), Source 2 (5,034,616), Source 3 (5,285,603), Ground Truth (2,206,821).
- Test: Source 1 (1,732,544), Source 2 (4,887,273), Source 3 (5,082,316).
- Format: Tab-separated (`sep="\t"`), 4 columns (`entity_id`, `business_name`, `business_address`, `country`), except ground truth which has 2 columns.
- Duplicate entity IDs: Exactly **0** in every file.

### B. Important Data-Quality Observations
- Reference Source 1 is pristine (0 missing values, clean English text).
- Sources 2 and 3 exhibit missing addresses in **2.65% to 3.36%** of records.
- Sources 2 and 3 contain systematic noise: leading punctuation symbols (`--`, `<<`), URL domains as company names (~4%), and regional Indian scripts (~9% in India records).
- Addresses in Sources 2 & 3 frequently invert hierarchical order (state or city listed first).

### C. Ground-Truth Matching Patterns
- **Total Ground-Truth Pairs:** 7,638,365 matched pairs (3,693,619 in S2, 3,944,746 in S3).
- **Cross-Source Synergy:** **80.48%** of S1 reference entities match records in **both** Source 2 and Source 3. Only 6.48% match S2 only, and 7.45% match S3 only.
- **Strict Country Isolation:** **100.00%** of matches occur within the same country.

### D. Singleton / Multiple-Match Observations
- **Singletons (0 matches):** **123,247 entities (5.58%)**.
- **Exactly 1 match:** **119,157 entities (5.40%)**.
- **Multiple matches (>1):** **1,964,417 entities (89.01%)**.
- Average matches per matched S1 entity is **3.66**, with a maximum of **11**.

### E. Preprocessing Requirements
- Systematic prefix noise stripping (`--`, `<<`).
- Domain extension stripping (`.com`, `.org`) to extract brand stems.
- Legal suffix canonical mapping across US (`LLC`, `Inc`), India (`Pvt Ltd`, `LLP`), and France (`SARL`, `SAS`).
- Indic script romanization / transliteration for Indian records.
- Address tokenization and numeric token extraction (house/plot numbers).

### F. Potential Blocking Strategies
- Level 1: Strict country partitioning (`US`, `India`, `France`).
- Level 2: Multi-pass union blocking combining:
  1. Normalized core name tokens.
  2. Alphanumeric domain/brand stem.
  3. Address number + location anchor.
  4. Character 3-gram / MinHash LSH retrieval.
- Top $k$ candidate selection ($k \approx 15\text{--}25$) to maintain high recall ceiling while producing a compact candidate space for `candidate_pairs.tsv`.

### G. Risks That Could Affect $F_{0.5}$
1. **False Merges on Generic Names:** Because 30.25% of S1 names are shared across distinct entities, false merges on common names will destroy precision. Since $F_{0.5}$ weights precision 2× over recall, address verification must gate matches.
2. **Singleton Penalty:** Predicting any false match on a singleton reduces its score from 1.0 to 0.0. The classification threshold must remain conservative.
3. **Out-of-Distribution Country (`France`):** Any country-specific heuristic tuned on US/India will fail on France. Pipeline components must remain country-aware and robust to open-ended countries.
4. **Missing Addresses in Candidate Generation:** 3.3% of S2/S3 records have no address; blocking strategies that strictly require address tokens will miss all of these records unless name-based passes are included.

---
*End of Phase 1 Report. Awaiting user instruction before proceeding to Phase 2.*
