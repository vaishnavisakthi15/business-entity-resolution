"""
Amazon ML Challenge 2026: Business Entity Resolution
Scalable Preprocessing Module

Provides high-throughput, memory-efficient normalization for:
1. Business Names (Unicode NFKD, accent stripping, prefix noise removal, URL domain extraction,
   legal suffix canonicalization, Indic transliteration).
2. Business Addresses (abbreviation expansion, component-inversion sorted tokens,
   numeric component extraction, null-safe handling).
3. Country Labels (open-ended internationalized string cleaning).
"""

import os
import sys
import re
import time
import unicodedata
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure stdout handles UTF-8 when printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. INDIC SCRIPT TRANSLITERATION DICTIONARIES
# ==============================================================================

DEVANAGARI_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'ऑ': 'o', 'ऍ': 'e',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    'ॉ': 'o', 'ॅ': 'e',
    '्': '', '़': '', '।': '.', '॥': '.'
}

DEVANAGARI_WORDS = {
    'प्राइवेट': 'pvt',
    'प्रा.': 'pvt',
    'लिमिटेड': 'ltd',
    'लि.': 'ltd',
    'एलएलपी': 'llp',
    'कंपनी': 'company',
    'मार्केटिंग': 'marketing',
    'प्रॉपर्टीज': 'properties',
    'फाइनेंस': 'finance',
    'इंटरप्राइजेज': 'enterprises',
    'ट्रेडर्स': 'traders',
    'सर्विसेज': 'services',
    'इंफोटेक': 'infotech',
    'एसोसिएट्स': 'associates'
}

TAMIL_MAP = {
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ee', 'உ': 'u', 'ஊ': 'oo',
    'எ': 'e', 'ஏ': 'ae', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oa', 'ஔ': 'au',
    'க': 'k', 'ங': 'ng', 'ச': 's', 'ஞ': 'ny', 'ட': 't', 'ண': 'n',
    'த': 'th', 'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r',
    'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l', 'ற': 'r', 'ன': 'n',
    'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h',
    'ா': 'a', 'ி': 'i', 'ீ': 'ee', 'ு': 'u', 'ூ': 'oo',
    'ெ': 'e', 'ே': 'ae', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oa', 'ௌ': 'au',
    '்': '', 'ஃ': 'k'
}

TAMIL_WORDS = {
    'இன்வெஸ்ட்மெண்ட்ஸ்': 'investments',
    'எல்எல்பி': 'llp',
    'பிரைவேட்': 'pvt',
    'லிமிடெட்': 'ltd',
    'தமிழ்நாடு': 'tamil nadu'
}

# ==============================================================================
# 2. LEGAL SUFFIXES & ABBREVIATION DICTIONARIES
# ==============================================================================

# Multijurisdictional legal suffixes (US, India, France)
LEGAL_SUFFIX_SET = {
    # US
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'pllc', 'lp', 'co', 'company',
    # India
    'pvt', 'private', 'ltd', 'limited', 'enterprises', 'traders',
    # France
    'sarl', 'sas', 'sasu', 'sci', 'eurl', 'sa', 'snc', 'gie', 'association', 'groupe', 'ste', 'cie'
}

# Standard address abbreviation expansions (avoiding state code collisions like CT, FL)
ADDRESS_ABBREV_MAP = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bdr\b': 'drive',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bbvd\b': 'boulevard',
    r'\bln\b': 'lane',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bbldg\b': 'building',
    r'\btq\b': 'taluka',
    r'\bdist\b': 'district',
    r'\bno\b': 'number',
    r'\br\.\b': 'rue',
    r'\brue\b': 'rue',
    r'\bbd\b': 'boulevard',
    r'\bav\b': 'avenue',
    r'\ball\b': 'allee',
    r'\bimp\b': 'impasse',
    r'\bpl\b': 'place',
}

# Regex to collapse acronyms with dots like L.L.C., S.A.S., C.I.T., U.S.
RE_ACRONYM_DOTS = re.compile(r'\b([a-zA-Z])\.([a-zA-Z])(?:\.([a-zA-Z]))?(?:\.([a-zA-Z]))?\.?\b')

def collapse_acronym_dots(text: str) -> str:
    """Collapses dot-separated acronyms (e.g., L.L.C. -> LLC, S.A.S. -> SAS)."""
    def _repl(m):
        groups = [g for g in m.groups() if g is not None]
        return "".join(groups)
    return RE_ACRONYM_DOTS.sub(_repl, text)

# Common address stop tokens that do not distinguish physical locations
ADDRESS_STOP_TOKENS = {
    'near', 'opp', 'opposite', 'behind', 'beside', 'floor', 'fl', 'suite', 'ste',
    'unit', 'apt', 'room', 'block', 'building', 'bldg', 'road', 'street', 'drive',
    'avenue', 'lane', 'court', 'boulevard', 'rue', 'allee', 'place', 'taluka', 'district'
}

# Regex pre-compiled for performance
RE_LEADING_GARBAGE = re.compile(r'^[-<>=*#@_~+/|!\?\^%]+')
RE_URL_PATTERN = re.compile(r'^(?:https?://)?(?:www\.)?([a-zA-Z0-9\-\.]+)\.(com|org|net|in|fr|co|io|biz|info|edu|gov)(?:/.*)?$', re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r'\.(com|org|net|in|fr|co|io|biz|info|edu|gov)\b', re.IGNORECASE)
RE_NON_ALPHANUM_SPACE = re.compile(r'[^a-zA-Z0-9\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+(?:[a-zA-Z]|\-\d+)?\b')

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def detect_script(text: str) -> str:
    """Detect script family of text."""
    if not text:
        return "empty"
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        if 0x0B80 <= code <= 0x0BFF:
            return "tamil"
        if (0x0980 <= code <= 0x0D7F):
            return "indic_other"
    return "latin"

def transliterate_indic(text: str) -> str:
    """Transliterate Indic (Devanagari, Tamil) text to phonetic Latin."""
    if not text:
        return ""
    res = text
    for word, repl in DEVANAGARI_WORDS.items():
        res = res.replace(word, repl)
    for word, repl in TAMIL_WORDS.items():
        res = res.replace(word, repl)
    
    out = []
    for ch in res:
        if ch in DEVANAGARI_MAP:
            out.append(DEVANAGARI_MAP[ch])
        elif ch in TAMIL_MAP:
            out.append(TAMIL_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)

def strip_accents(text: str) -> str:
    """Unicode NFKD decomposition to strip accents (e.g., é -> e)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def extract_url_domain_root(text: str) -> Tuple[bool, str]:
    """Check if name is a URL / website domain and extract the clean brand root."""
    m = RE_URL_PATTERN.match(text.strip())
    if m:
        domain = m.group(1).lower()
        # Remove subdomains like 'shop.' or 'app.'
        parts = domain.split('.')
        root = parts[-1] if len(parts) > 1 else parts[0]
        # Replace hyphens with spaces
        clean_root = root.replace('-', ' ')
        return True, clean_root
    return False, ""

# ==============================================================================
# 4. CORE NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_country(country: Optional[str]) -> str:
    """Normalize country label without hardcoding country list."""
    if country is None:
        return "UNKNOWN"
    c = str(country).strip()
    return c.upper() if c else "UNKNOWN"

def normalize_name(raw_name: Optional[str]) -> Dict[str, Any]:
    """
    Normalize business name while preserving original values and creating:
    - name_normalized: lowercase, cleaned punctuation, NFKD accent-stripped.
    - name_tokens: list of all alphanumeric tokens.
    - name_significant_tokens: list of tokens excluding common legal suffixes.
    - name_brand_root: core alphanumeric brand or domain root.
    - is_url: boolean indicator.
    - name_script: script classification.
    - name_transliterated: phonetic Latin representation if non-Latin.
    """
    raw = "" if raw_name is None else str(raw_name).strip()
    if not raw or raw.lower() == "nan":
        return {
            "name_normalized": "",
            "name_tokens": [],
            "name_significant_tokens": [],
            "name_brand_root": "",
            "is_url": False,
            "name_script": "empty",
            "name_transliterated": ""
        }
    
    script = detect_script(raw)
    
    # Transliteration if Indic script
    transliterated = ""
    work_text = raw
    if script in ("devanagari", "tamil", "indic_other"):
        transliterated = transliterate_indic(raw)
        work_text = transliterated
        
    # Check if URL
    is_url, domain_root = extract_url_domain_root(work_text)
    if is_url:
        work_text = domain_root
    else:
        # Strip domain extension if present inside text
        work_text = RE_DOMAIN_SUFFIX.sub('', work_text)
        
    # Strip leading noise symbols (-- , << , >> , etc.)
    work_text = RE_LEADING_GARBAGE.sub('', work_text).strip()
    
    # Strip accents
    work_text = strip_accents(work_text)
    
    # Lowercase
    work_text = work_text.lower()
    
    # Collapse acronym dots (e.g. l.l.c. -> llc, s.a.s. -> sas)
    work_text = collapse_acronym_dots(work_text)
    
    # Clean punctuation to space, keeping alphanumerics
    cleaned = RE_NON_ALPHANUM_SPACE.sub(' ', work_text)
    cleaned = RE_MULTI_SPACE.sub(' ', cleaned).strip()
    
    # Tokenize
    tokens = cleaned.split() if cleaned else []
    
    # Significant tokens (exclude legal suffixes)
    sig_tokens = [t for t in tokens if t not in LEGAL_SUFFIX_SET]
    if not sig_tokens and tokens:
        # If all tokens were legal words (e.g. "The Group LLC"), retain original
        sig_tokens = tokens
        
    # Brand root
    brand_root = "".join(sig_tokens) if sig_tokens else ("".join(tokens) if tokens else "")
    
    return {
        "name_normalized": cleaned,
        "name_tokens": tokens,
        "name_significant_tokens": sig_tokens,
        "name_brand_root": brand_root,
        "is_url": is_url,
        "name_script": script,
        "name_transliterated": transliterated
    }

def normalize_address(raw_address: Optional[str]) -> Dict[str, Any]:
    """
    Normalize business address:
    - address_normalized: lowercased, expanded abbreviations, accent-stripped.
    - address_tokens: list of tokens.
    - address_sorted_tokens: sorted unique significant tokens (order-invariant).
    - address_numbers: extracted numeric components (building, plot, postal code).
    - has_address: boolean flag.
    """
    raw = "" if raw_address is None else str(raw_address).strip()
    if not raw or raw.lower() == "nan":
        return {
            "address_normalized": "",
            "address_tokens": [],
            "address_sorted_tokens": [],
            "address_numbers": [],
            "has_address": False
        }
        
    script = detect_script(raw)
    work_text = raw
    if script in ("devanagari", "tamil", "indic_other"):
        work_text = transliterate_indic(raw)
        
    # Strip accents
    work_text = strip_accents(work_text)
    work_text = work_text.lower()
    
    # Collapse acronym dots (e.g. c.i.t. -> cit, n.c. -> nc)
    work_text = collapse_acronym_dots(work_text)
    
    # Expand common abbreviations
    for pattern, repl in ADDRESS_ABBREV_MAP.items():
        work_text = re.sub(pattern, repl, work_text)
        
    # Extract numbers before removing punctuation
    numbers = RE_NUMBERS.findall(work_text)
    
    # Clean non-alphanumeric characters
    cleaned = RE_NON_ALPHANUM_SPACE.sub(' ', work_text)
    cleaned = RE_MULTI_SPACE.sub(' ', cleaned).strip()
    
    tokens = cleaned.split() if cleaned else []
    
    # Filter out common address stopwords for the sorted signature
    sig_tokens = [t for t in tokens if t not in ADDRESS_STOP_TOKENS]
    sorted_unique_tokens = sorted(set(sig_tokens)) if sig_tokens else sorted(set(tokens))
    
    return {
        "address_normalized": cleaned,
        "address_tokens": tokens,
        "address_sorted_tokens": sorted_unique_tokens,
        "address_numbers": numbers,
        "has_address": True
    }

# ==============================================================================
# 5. HIGH-THROUGHPUT BATCH PROCESSOR
# ==============================================================================

def preprocess_dataframe_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scalable vector-friendly batch preprocessing on a DataFrame chunk.
    Preserves all original columns and adds separate normalized columns.
    """
    n_rows = len(df)
    
    norm_names = [None] * n_rows
    sig_name_tokens_str = [None] * n_rows
    brand_roots = [None] * n_rows
    is_urls = [False] * n_rows
    name_scripts = [None] * n_rows
    
    norm_addrs = [None] * n_rows
    sorted_addr_tokens_str = [None] * n_rows
    addr_numbers_str = [None] * n_rows
    has_addresses = [False] * n_rows
    
    norm_countries = [None] * n_rows
    
    names_raw = df["business_name"].values
    addrs_raw = df["business_address"].values
    ctrys_raw = df["country"].values
    
    for i in range(n_rows):
        # Country
        norm_countries[i] = normalize_country(ctrys_raw[i])
        
        # Name
        n_res = normalize_name(names_raw[i])
        norm_names[i] = n_res["name_normalized"]
        sig_name_tokens_str[i] = " ".join(n_res["name_significant_tokens"])
        brand_roots[i] = n_res["name_brand_root"]
        is_urls[i] = n_res["is_url"]
        name_scripts[i] = n_res["name_script"]
        
        # Address
        a_res = normalize_address(addrs_raw[i])
        norm_addrs[i] = a_res["address_normalized"]
        sorted_addr_tokens_str[i] = " ".join(a_res["address_sorted_tokens"])
        addr_numbers_str[i] = " ".join(a_res["address_numbers"])
        has_addresses[i] = a_res["has_address"]
        
    out_df = df.copy()
    out_df["country_normalized"] = norm_countries
    out_df["name_normalized"] = norm_names
    out_df["name_significant_tokens"] = sig_name_tokens_str
    out_df["name_brand_root"] = brand_roots
    out_df["is_url"] = is_urls
    out_df["name_script"] = name_scripts
    
    out_df["address_normalized"] = norm_addrs
    out_df["address_sorted_tokens"] = sorted_addr_tokens_str
    out_df["address_numbers"] = addr_numbers_str
    out_df["has_address"] = has_addresses
    
    return out_df

def preprocess_tsv_to_parquet(
    input_tsv_path: str,
    output_parquet_path: str,
    chunksize: int = 250000,
    max_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Streamingly preprocess a large TSV file chunk-by-chunk and write to Parquet.
    Uses constant memory, reports progress, and records processing statistics.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import psutil
    
    t0 = time.time()
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    
    total_processed = 0
    writer = None
    
    print(f"Starting preprocessing of {os.path.basename(input_tsv_path)} -> {os.path.basename(output_parquet_path)}")
    
    for chunk in pd.read_csv(
        input_tsv_path,
        sep="\t",
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        nrows=max_rows
    ):
        processed_chunk = preprocess_dataframe_chunk(chunk)
        
        table = pa.Table.from_pandas(processed_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_parquet_path, table.schema, compression="snappy")
        writer.write_table(table)
        
        total_processed += len(chunk)
        mem = psutil.virtual_memory()
        print(f"  Processed {total_processed:,} rows (RAM: {mem.used/(1024**3):.2f} GB / {mem.total/(1024**3):.2f} GB)")
        
    if writer is not None:
        writer.close()
        
    dt = time.time() - t0
    size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)
    print(f"Finished {total_processed:,} rows in {dt:.2f}s ({total_processed/dt:.1f} rows/s). Parquet size: {size_mb:.2f} MB")
    
    return {
        "input_file": input_tsv_path,
        "output_file": output_parquet_path,
        "rows_processed": total_processed,
        "elapsed_seconds": round(dt, 2),
        "rows_per_second": round(total_processed / dt, 1),
        "parquet_size_mb": round(size_mb, 2)
    }

if __name__ == "__main__":
    print("Testing preprocessing module on edge cases...")
    test_names = [
        "-- Holloway Peak Inc Seafood",
        "<< Team Ecole",
        "maurewilliamscolombier.com",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "Crystal Staffing Solutions LLC",
        "Obsidian, LLC",
        "Fractales Amis Groupe S.A.S"
    ]
    for n in test_names:
        print(f"\nRAW: {n}")
        res = normalize_name(n)
        print(f" -> NORM:        {res['name_normalized']}")
        print(f" -> SIG TOKENS:  {res['name_significant_tokens']}")
        print(f" -> BRAND ROOT:  {res['name_brand_root']}")
        print(f" -> SCRIPT:      {res['name_script']}")
        if res['name_transliterated']:
            print(f" -> TRANSLIT:    {res['name_transliterated']}")
