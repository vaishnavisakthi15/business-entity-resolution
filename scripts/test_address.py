import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import normalize_address

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

test_addrs = [
    "33 Sleepy Hollow Drive, Danbury, CT",
    "CT, SLEEPY HOLLOW DRIVE, DANBURY",
    "8706 KENTUCKY DERBY DR, WAXHAW, NC",
    "8706 Kentucky Derby Drive, Waxhaw, NC",
    "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
    "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
    "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
    "",
    "nan"
]

for a in test_addrs:
    print(f"\nRAW: {a}")
    res = normalize_address(a)
    print(f" -> NORM:         {res['address_normalized']}")
    print(f" -> SORTED TOK:   {res['address_sorted_tokens']}")
    print(f" -> NUMBERS:      {res['address_numbers']}")
    print(f" -> HAS ADDRESS:  {res['has_address']}")
