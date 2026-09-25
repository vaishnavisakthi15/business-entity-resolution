import os
import sys
import json
import re
from collections import Counter
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"

# We'll analyze a representative sample of 100k rows from train_source1, train_source2, test_source1, test_source2
datasets_to_check = [
    ("train_source1", os.path.join(BASE_DIR, "train", "train_source1.tsv"), 100000),
    ("train_source2", os.path.join(BASE_DIR, "train", "train_source2.tsv"), 100000),
    ("train_source3", os.path.join(BASE_DIR, "train", "train_source3.tsv"), 100000),
    ("test_source1", os.path.join(BASE_DIR, "test", "test_source1.tsv"), 100000),
    ("test_source2", os.path.join(BASE_DIR, "test", "test_source2.tsv"), 100000),
    ("test_source3", os.path.join(BASE_DIR, "test", "test_source3.tsv"), 100000),
]

pattern_results = {}

for name, path, sample_size in datasets_to_check:
    print(f"Analyzing patterns in {name} (sample size: {sample_size})...")
    df = pd.read_csv(path, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
    
    leading_noise_cnt = 0
    non_ascii_name_cnt = 0
    devanagari_cnt = 0
    tamil_cnt = 0
    other_indic_cnt = 0
    accented_latin_cnt = 0
    url_name_cnt = 0
    
    has_digits_addr_cnt = 0
    pin_or_zip_cnt = 0
    empty_addr_cnt = 0
    
    punctuation_counter = Counter()
    common_legal_tokens = Counter()
    
    for b_name, b_addr, ctry in zip(df["business_name"], df["business_address"], df["country"]):
        b_name = b_name.strip()
        b_addr = b_addr.strip()
        
        # Punctuation check in name
        for ch in b_name:
            if not ch.isalnum() and not ch.isspace():
                punctuation_counter[ch] += 1
                
        # Leading noise symbols
        if re.match(r'^[-<>=*#@_~+/]{2,}', b_name):
            leading_noise_cnt += 1
            
        # URL detection
        if re.search(r'\.(com|org|net|in|fr|co|io)\b', b_name.lower()):
            url_name_cnt += 1
            
        # Script check
        has_non_ascii = any(ord(c) > 127 for c in b_name)
        if has_non_ascii:
            non_ascii_name_cnt += 1
            if any('\u0900' <= c <= '\u097F' for c in b_name):
                devanagari_cnt += 1
            elif any('\u0B80' <= c <= '\u0BFF' for c in b_name):
                tamil_cnt += 1
            elif any(('\u0980' <= c <= '\u0D7F') for c in b_name):
                other_indic_cnt += 1
            else:
                accented_latin_cnt += 1
                
        # Address patterns
        if not b_addr or b_addr.lower() == "nan":
            empty_addr_cnt += 1
        else:
            if any(c.isdigit() for c in b_addr):
                has_digits_addr_cnt += 1
            # 5 or 6 digit postal code
            if re.search(r'\b\d{5,6}\b', b_addr):
                pin_or_zip_cnt += 1
                
        # Legal tokens
        tokens = re.findall(r'[a-zA-Z]+', b_name.lower())
        for t in ["inc", "llc", "corp", "corporation", "ltd", "limited", "pvt", "private", "llp", "sarl", "sas", "sci", "eurl", "sa"]:
            if t in tokens:
                common_legal_tokens[t] += 1

    pattern_results[name] = {
        "sample_size": sample_size,
        "leading_noise_symbols_count": leading_noise_cnt,
        "url_as_name_count": url_name_cnt,
        "non_ascii_name_count": non_ascii_name_cnt,
        "devanagari_script_count": devanagari_cnt,
        "tamil_script_count": tamil_cnt,
        "other_indic_script_count": other_indic_cnt,
        "accented_latin_count": accented_latin_cnt,
        "empty_address_count": empty_addr_cnt,
        "address_has_digits_count": has_digits_addr_cnt,
        "address_has_5_or_6_digit_code": pin_or_zip_cnt,
        "top_punctuation": dict(punctuation_counter.most_common(10)),
        "common_legal_tokens": dict(common_legal_tokens.most_common(10)),
    }

with open(r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\pattern_analysis.json", "w", encoding="utf-8") as f:
    json.dump(pattern_results, f, indent=2, ensure_ascii=False)

print("Saved pattern_analysis.json")
