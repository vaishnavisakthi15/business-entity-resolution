import unicodedata
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Comprehensive Devanagari mapping
DEVANAGARI_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    '्': '', '़': '', '।': '.', '॥': '.'
}

# Devanagari legal/business vocabulary overrides
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

# Tamil mapping
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

def transliterate_indic(text: str) -> str:
    # First replace whole known words
    res = text
    for word, repl in DEVANAGARI_WORDS.items():
        res = res.replace(word, repl)
    for word, repl in TAMIL_WORDS.items():
        res = res.replace(word, repl)
    
    # Character by character transliteration for remaining Indic chars
    out = []
    for ch in res:
        if ch in DEVANAGARI_MAP:
            out.append(DEVANAGARI_MAP[ch])
        elif ch in TAMIL_MAP:
            out.append(TAMIL_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)

test_cases = [
    'राम मार्केटिंग प्राइवेट लिमिटेड',
    'आदित्य प्रॉपर्टीज एलएलपी',
    'मॉडर्न फाइनेंस',
    'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி',
    'Raj Investments எல்எல்பி',
    'தமிழ்நாடு'
]

for tc in test_cases:
    print(f"{tc:35s} -> {transliterate_indic(tc)}")
