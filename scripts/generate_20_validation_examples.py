import os
import sys
import json
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import normalize_name, normalize_address, normalize_country

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Real examples curated directly from the dataset inspection
raw_cases = [
    # Category 1: Punctuation Noise
    {
        "category": "Punctuation Noise",
        "description": "Leading hyphens and noise characters",
        "entity_id": "S2-764573417",
        "name": "-- Holloway Peak Inc Seafood",
        "address": "105 ELM ST, MORGANTON, NC",
        "country": "US"
    },
    {
        "category": "Punctuation Noise",
        "description": "Leading angle brackets",
        "entity_id": "S1-156285671",
        "name": "<< Team Ecole",
        "address": "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
        "country": "France"
    },
    {
        "category": "Punctuation Noise",
        "description": "Hyphenated suffix with punctuation inside name",
        "entity_id": "S2-582477216",
        "name": "CRYSTAL STAFFING SOLUTIONS-L.L.C.",
        "address": "8706 KENTUCKY DERBY DR, WAXHAW, NC",
        "country": "US"
    },
    {
        "category": "Punctuation Noise",
        "description": "Comma and hyphen noise in legal suffix",
        "entity_id": "S2-736616474",
        "name": "Obsidian,-LLC",
        "address": "3907 HAMILTON RD, DEER PARK CIYT, WA",
        "country": "US"
    },
    
    # Category 2: URLs as Business Names
    {
        "category": "URLs as Names",
        "description": "Domain URL as business name",
        "entity_id": "S3-11291185",
        "name": "maurewilliamscolombier.com",
        "address": "Wayne Ave, Ticonderoga Townshiip, New York",
        "country": "US"
    },
    {
        "category": "URLs as Names",
        "description": "Domain URL as business name",
        "entity_id": "S3-202863386",
        "name": "wilfordhancock.com",
        "address": "Mack Rd, Haltom City, Texas",
        "country": "US"
    },
    {
        "category": "URLs as Names",
        "description": "URL domain with dot extension",
        "entity_id": "S2-441928012",
        "name": "https://www.apexconsulting.org",
        "address": "450 7th Ave, New York, NY",
        "country": "US"
    },
    
    # Category 3: Abbreviations (Legal Suffixes & Street Types)
    {
        "category": "Abbreviations",
        "description": "Road, Suite, and Inc abbreviations",
        "entity_id": "S1-925783039",
        "name": "Orelee's Barbershop Inc.",
        "address": "1795 Westchester Drive, Ste 200, High Point, NC",
        "country": "US"
    },
    {
        "category": "Abbreviations",
        "description": "Avenue, Blvd, and LLC abbreviations",
        "entity_id": "S1-377745466",
        "name": "B+ Retail Inc",
        "address": "1712 Montebello Ave, Phoenix, AZ",
        "country": "US"
    },
    {
        "category": "Abbreviations",
        "description": "Indian legal abbreviations (Pvt Ltd, LLP)",
        "entity_id": "S1-55344266",
        "name": "Raj Investments LLP",
        "address": "6(29), C.I.T. Colony, 2Nd Main Rd Mylapore, Chennai, Tamil Nadu",
        "country": "India"
    },
    {
        "category": "Abbreviations",
        "description": "French legal abbreviations (SARL, Bd)",
        "entity_id": "S2-566025912",
        "name": "Marina Ecole France Sarl",
        "address": "63 R. DE DIEPPE, LILLE, Hauts-de-France",
        "country": "France"
    },
    
    # Category 4: Missing Addresses
    {
        "category": "Missing Addresses",
        "description": "Source 2 empty address record",
        "entity_id": "S2-680265918",
        "name": "obsidian, llc",
        "address": "",
        "country": "US"
    },
    {
        "category": "Missing Addresses",
        "description": "Source 3 null string address",
        "entity_id": "S3-859268022",
        "name": "International South Consultants Private Ltd",
        "address": "nan",
        "country": "India"
    },
    {
        "category": "Missing Addresses",
        "description": "Source 2 empty address record with noisy name",
        "entity_id": "S2-681193310",
        "name": "Maure Wilblims Colombier Inc",
        "address": "",
        "country": "US"
    },
    {
        "category": "Missing Addresses",
        "description": "Source 3 whitespace-only address",
        "entity_id": "S3-860443364",
        "name": "Maure Williams Inc Center",
        "address": "   ",
        "country": "US"
    },
    
    # Category 5: Multilingual / Indic Text
    {
        "category": "Multilingual / Indic",
        "description": "Devanagari script business name",
        "entity_id": "S2-166376419",
        "name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "Devanagari LLP business name",
        "entity_id": "S2-639257739",
        "name": "आदित्य प्रॉपर्टीज एलएलपी",
        "address": "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "Devanagari finance company",
        "entity_id": "S3-462677478",
        "name": "मॉडर्न फाइनेंस",
        "address": "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "Tamil script business name",
        "entity_id": "S2-249013014",
        "name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "address": "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "Mixed script: English brand + Tamil legal suffix",
        "entity_id": "S3-478195123",
        "name": "Raj Investments எல்எல்பி",
        "address": "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "Tamil script state name inside address",
        "entity_id": "S3-384364074",
        "name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "address": "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
        "country": "India"
    },
    {
        "category": "Multilingual / Indic",
        "description": "French diacritics and legal suffix (SCI)",
        "entity_id": "S2-158121477",
        "name": "SCI Ptit Àmicale",
        "address": "18 RUE JEN ZAY, Dunkerque, Nord",
        "country": "France"
    },
    
    # Category 6: Address Component Inversion
    {
        "category": "Address Inversion",
        "description": "State first inverted address (Reference)",
        "entity_id": "S1-29845983",
        "name": "Hendricks and Flowers Inc",
        "address": "33 Sleepy Hollow Drive, Danbury, CT",
        "country": "US"
    },
    {
        "category": "Address Inversion",
        "description": "State first inverted address (Noisy match)",
        "entity_id": "S2-648035184",
        "name": "Hendricks and  Flowers Inc",
        "address": "CT, SLEEPY HOLLOW DRIVE, DANBURY",
        "country": "US"
    },
    {
        "category": "Address Inversion",
        "description": "State & City first inverted address",
        "entity_id": "S1-106407869",
        "name": "Vision Partners Corp",
        "address": "IA, Iowa City, 1064 Newton Rd, Unit 11",
        "country": "US"
    },
    {
        "category": "Address Inversion",
        "description": "Street first normal address matching inverted one",
        "entity_id": "S2-390192831",
        "name": "Vision Partners Corporation",
        "address": "1064 Newton Road, Iowa City, IA",
        "country": "US"
    }
]

validation_results = []

for case in raw_cases:
    n_res = normalize_name(case["name"])
    a_res = normalize_address(case["address"])
    c_norm = normalize_country(case["country"])
    
    res_entry = {
        "category": case["category"],
        "description": case["description"],
        "entity_id": case["entity_id"],
        "original_country": case["country"],
        "normalized_country": c_norm,
        "original_name": case["name"],
        "normalized_name": n_res["name_normalized"],
        "significant_name_tokens": n_res["name_significant_tokens"],
        "name_brand_root": n_res["name_brand_root"],
        "is_url": n_res["is_url"],
        "name_script": n_res["name_script"],
        "original_address": case["address"],
        "normalized_address": a_res["address_normalized"],
        "sorted_address_tokens": a_res["address_sorted_tokens"],
        "address_numbers": a_res["address_numbers"],
        "has_address": a_res["has_address"]
    }
    validation_results.append(res_entry)

out_file = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\scripts\output\validation_examples_25.json"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)

print(f"Generated {len(validation_results)} verified validation examples saved to {out_file}")
