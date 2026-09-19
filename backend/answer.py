"""
BIS Sahayak Answer Generator.

Answers Indian Standards & BIS queries using live AI reasoning,
with a robust fallback to the local verified BIS dataset (knowledge_base.json).

Modes:
  auto     - Default. Tries AI engine first for dynamic, comprehensive BIS
             intelligence. If no API key is provided, or the network is down,
             or an error occurs, it seamlessly falls back to the
             local verified knowledge base.
  groq     - Forces AI engine response.
  dataset  - Forces local BM25 + knowledge_base.json retrieval (offline mode).
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Automatically load .env file from project root if present (zero external deps)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")


def load_env_file():
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


load_env_file()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL_ANSWER", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

NO_ANSWER = {
    "en": (
        "I could not find this in the BIS sources loaded into me, so I am not "
        "going to guess. Try naming the product and what it is made of, or ask "
        "about a scheme by name (ISI, CRS, hallmarking, FMCS)."
    ),
    "hi": (
        "यह जानकारी मेरे पास उपलब्ध BIS स्रोतों में नहीं मिली, इसलिए मैं अनुमान "
        "नहीं लगाऊँगा। कृपया उत्पाद का नाम और सामग्री बताएँ, या किसी योजना का नाम "
        "लिखें (ISI, CRS, हॉलमार्किंग, FMCS)।"
    ),
}

REFUSAL_TERMS = [
    "[REFUSAL]",
    "cannot answer unrelated",
    "असंबंधित प्रश्नों का उत्तर नहीं",
    "outside the scope of bis",
    "specialized only in indian standards",
    "not within the purview of bis",
    "unrelated to bis",
    "does not fall under bis",
    "cannot provide recommendations for pizza",
    "cannot provide information on cricket",
    "cannot provide advice on cryptocurrency",
    "exclusively dedicated to indian standards",
]


def get_all_groq_keys(explicit_key=None):
    """
    Returns an ordered, deduplicated list of available Groq API keys.
    Supports:
      1. Explicit keys from caller (single or comma-separated)
      2. GROQ_API_KEYS (comma/semicolon/newline/pipe separated)
      3. GROQ_API_KEY (single or comma-separated)
      4. Numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3...
      5. Fallback aliases: GROK_API_KEY*, XAI_API_KEY*
    """
    keys = []

    def add_key_str(raw):
        if not raw:
            return
        parts = re.split(r"[,;\n\|]+", str(raw))
        for p in parts:
            p = p.strip().strip("'\"")
            if p and p not in keys:
                keys.append(p)

    # 1. Explicit key(s) passed in call or HTTP header
    if explicit_key:
        add_key_str(explicit_key)

    # 2. Plural environment variables
    add_key_str(os.environ.get("GROQ_API_KEYS"))
    add_key_str(os.environ.get("GROK_API_KEYS"))

    # 3. Standard environment variables
    add_key_str(os.environ.get("GROQ_API_KEY"))
    add_key_str(os.environ.get("GROK_API_KEY"))
    add_key_str(os.environ.get("XAI_API_KEY"))

    # 4. Numbered environment variables: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
    numbered = []
    for k, v in os.environ.items():
        m = re.match(r"^GROQ_API_KEY_(\d+)$", k, re.IGNORECASE) or re.match(
            r"^GROK_API_KEY_(\d+)$", k, re.IGNORECASE
        )
        if m:
            numbered.append((int(m.group(1)), v))
    for _, val in sorted(numbered, key=lambda x: x[0]):
        add_key_str(val)

    return keys


def get_groq_api_key(explicit_key=None):
    """Returns the primary Groq API key (or first available key in pool)."""
    keys = get_all_groq_keys(explicit_key)
    return keys[0] if keys else ""


def get_groq_model():
    return (
        os.environ.get("GROQ_MODEL")
        or os.environ.get("GROK_MODEL")
        or DEFAULT_GROQ_MODEL
    ).strip()


def is_groq_configured(explicit_key=None):
    return len(get_all_groq_keys(explicit_key)) > 0


DEFAULT_PORTAL_URL = (
    "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/indian_standards/isdetails"
)

STANDARDS_DOC_MAP = {
    # Footwear & Personal Protective Equipment
    "IS 15298 (Part 2)": {
        "url": "https://archive.org/details/gov.in.is.15298.2.2011",
        "title": "BIS Official Standard - IS 15298 Part 2 (Safety Footwear)",
    },
    "IS 15298 (Part 3)": {
        "url": "https://archive.org/details/gov.in.is.15298.3.2011",
        "title": "BIS Official Standard - IS 15298 Part 3 (Protective Footwear)",
    },
    "IS 15298 (Part 4)": {
        "url": "https://archive.org/details/gov.in.is.15298.4.2010",
        "title": "BIS Official Standard - IS 15298 Part 4 (Occupational Footwear)",
    },
    "IS 15298": {
        "url": "https://archive.org/details/gov.in.is.15298.2.2011",
        "title": "BIS Official Standard - IS 15298 (Safety & Protective Footwear)",
    },
    "IS 6721": {
        "url": "https://archive.org/details/gov.in.is.6721.2023",
        "title": "BIS Official Standard - IS 6721 : 2023 (PVC Footwear & Sandals)",
    },
    "IS 10702": {
        "url": "https://archive.org/details/gov.in.is.10702.2023",
        "title": "BIS Official Standard - IS 10702 : 2023 (Rubber Hawai Chappals)",
    },
    "IS 15844": {
        "url": "https://archive.org/details/gov.in.is.15844.2010",
        "title": "BIS Official Standard - IS 15844 (Sports Footwear)",
    },
    "IS 11544": {
        "url": "https://archive.org/details/gov.in.is.11544.1986",
        "title": "BIS Official Standard - IS 11544 (Leather School Shoes)",
    },
    "IS 4151": {
        "url": "https://archive.org/details/gov.in.is.4151.2015",
        "title": "BIS Official Standard - IS 4151 : 2015 (Motorcycle Helmets)",
    },
    "IS 2925": {
        "url": "https://archive.org/details/gov.in.is.2925.1984",
        "title": "BIS Official Standard - IS 2925 (Industrial Safety Helmets)",
    },
    "IS 5983": {
        "url": "https://archive.org/details/gov.in.is.5983.1980",
        "title": "BIS Official Standard - IS 5983 (Eye-Protectors / Sunglasses)",
    },
    "IS 16289": {
        "url": "https://archive.org/details/gov.in.is.16289.2014",
        "title": "BIS Official Standard - IS 16289 (Medical Face Masks)",
    },

    # Construction & Steel & Cement
    "IS 1786": {
        "url": "https://archive.org/details/gov.in.is.1786.2008",
        "title": "BIS Official Standard - IS 1786 : 2008 (TMT Steel Rebars)",
    },
    "IS 456": {
        "url": "https://archive.org/details/gov.in.is.456.2000",
        "title": "BIS Official Standard - IS 456 : 2000 (Plain & Reinforced Concrete)",
    },
    "IS 1489 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.1489.1.1991",
        "title": "BIS Official Standard - IS 1489 Part 1 (PPC Cement)",
    },
    "IS 1489": {
        "url": "https://archive.org/details/gov.in.is.1489.1.1991",
        "title": "BIS Official Standard - IS 1489 Part 1 (PPC Cement)",
    },
    "IS 12269": {
        "url": "https://archive.org/details/gov.in.is.12269.2013",
        "title": "BIS Official Standard - IS 12269 : 2013 (OPC 53 Cement)",
    },
    "IS 269": {
        "url": "https://archive.org/details/gov.in.is.269.2015",
        "title": "BIS Official Standard - IS 269 (Ordinary Portland Cement)",
    },
    "IS 383": {
        "url": "https://archive.org/details/gov.in.is.383.2016",
        "title": "BIS Official Standard - IS 383 (Aggregates for Concrete)",
    },
    "IS 2062": {
        "url": "https://archive.org/details/gov.in.is.2062.2011",
        "title": "BIS Official Standard - IS 2062 (Structural Steel)",
    },
    "IS 303": {
        "url": "https://archive.org/details/gov.in.is.303.1989",
        "title": "BIS Official Standard - IS 303 (Plywood MR & BWR)",
    },
    "IS 814": {
        "url": "https://archive.org/details/gov.in.is.814.2004",
        "title": "BIS Official Standard - IS 814 (Covered Welding Electrodes)",
    },

    # Fire Safety & Domestic Gas
    "IS 15683": {
        "url": "https://archive.org/details/gov.in.is.15683.2018",
        "title": "BIS Official Standard - IS 15683 : 2018 (Portable Fire Extinguishers)",
    },
    "IS 3196 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.3196.1.2006",
        "title": "BIS Official Standard - IS 3196 Part 1 (LPG Cylinders)",
    },
    "IS 3196": {
        "url": "https://archive.org/details/gov.in.is.3196.1.2006",
        "title": "BIS Official Standard - IS 3196 Part 1 (LPG Cylinders)",
    },
    "IS 2347": {
        "url": "https://archive.org/details/gov.in.is.2347.2006",
        "title": "BIS Official Standard - IS 2347 (Domestic Pressure Cookers)",
    },
    "IS 4246": {
        "url": "https://archive.org/details/gov.in.is.4246.2002",
        "title": "BIS Official Standard - IS 4246 (LPG Domestic Gas Stoves)",
    },

    # Electrical, Electronics & Lighting
    "IS 302 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.302.1.2008",
        "title": "BIS Official Standard - IS 302 Part 1 (Electrical Appliances Safety)",
    },
    "IS 302 (Part 2/Sec 3)": {
        "url": "https://archive.org/details/gov.in.is.302.2.3.2007",
        "title": "BIS Official Standard - IS 302 (Part 2/Sec 3) (Electric Irons)",
    },
    "IS 302 (Part 2/Sec 14)": {
        "url": "https://archive.org/details/gov.in.is.302.2.14.2009",
        "title": "BIS Official Standard - IS 302 (Part 2/Sec 14) (Mixers & Grinders)",
    },
    "IS 302": {
        "url": "https://archive.org/details/gov.in.is.302.1.2008",
        "title": "BIS Official Standard - IS 302 (Electrical Appliances Safety)",
    },
    "IS 694": {
        "url": "https://archive.org/details/gov.in.is.694.2010",
        "title": "BIS Official Standard - IS 694 : 2010 (PVC Electric Cables)",
    },
    "IS 1293": {
        "url": "https://archive.org/details/gov.in.is.1293.2005",
        "title": "BIS Official Standard - IS 1293 (Plugs and Socket-Outlets)",
    },
    "IS 374": {
        "url": "https://archive.org/details/gov.in.is.374.1979",
        "title": "BIS Official Standard - IS 374 (Electric Ceiling Fans)",
    },
    "IS 2082": {
        "url": "https://archive.org/details/gov.in.is.2082.1993",
        "title": "BIS Official Standard - IS 2082 (Electric Water Heaters / Geysers)",
    },
    "IS 16102 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.16102.1.2012",
        "title": "BIS Official Standard - IS 16102 Part 1 (Self-Ballasted LED Lamps)",
    },
    "IS 16102": {
        "url": "https://archive.org/details/gov.in.is.16102.1.2012",
        "title": "BIS Official Standard - IS 16102 (LED Bulbs)",
    },
    "IS 13252 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.13252.1.2010",
        "title": "BIS Official Standard - IS 13252 Part 1 (IT Equipment Safety)",
    },
    "IS 13252": {
        "url": "https://archive.org/details/gov.in.is.13252.1.2010",
        "title": "BIS Official Standard - IS 13252 (IT Equipment Safety)",
    },
    "IS 16046 (Part 1 & 2)": {
        "url": "https://archive.org/details/gov.in.is.16046.1.2018",
        "title": "BIS Official Standard - IS 16046 (Secondary Lithium-ion Batteries)",
    },
    "IS 16046": {
        "url": "https://archive.org/details/gov.in.is.16046.1.2018",
        "title": "BIS Official Standard - IS 16046 (Lithium-ion Batteries)",
    },
    "IS 14286": {
        "url": "https://archive.org/details/gov.in.is.14286.2010",
        "title": "BIS Official Standard - IS 14286 (Solar PV Modules)",
    },

    # Food, Water & Medical
    "IS 14543": {
        "url": "https://archive.org/details/gov.in.is.14543.2016",
        "title": "BIS Official Standard - IS 14543 : 2016 (Packaged Drinking Water)",
    },
    "IS 13428": {
        "url": "https://archive.org/details/gov.in.is.13428.2005",
        "title": "BIS Official Standard - IS 13428 (Packaged Natural Mineral Water)",
    },
    "IS 10500": {
        "url": "https://archive.org/details/gov.in.is.10500.2012",
        "title": "BIS Official Standard - IS 10500 : 2012 (Drinking Water)",
    },
    "IS 3025": {
        "url": "https://archive.org/details/gov.in.is.3025.1.1987",
        "title": "BIS Official Standard - IS 3025 (Water Sampling & Test Methods)",
    },
    "IS 10146": {
        "url": "https://archive.org/details/gov.in.is.10146.1982",
        "title": "BIS Official Standard - IS 10146 (Polyethylene for Food Contact)",
    },
    "IS 1165": {
        "url": "https://archive.org/details/gov.in.is.1165.2002",
        "title": "BIS Official Standard - IS 1165 (Milk Powder Specification)",
    },
    "IS 3055 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.3055.1.1994",
        "title": "BIS Official Standard - IS 3055 Part 1 (Clinical Thermometers)",
    },
    "IS 3055": {
        "url": "https://archive.org/details/gov.in.is.3055.1.1994",
        "title": "BIS Official Standard - IS 3055 (Clinical Thermometers)",
    },

    # Toys & Automotive
    "IS 9873 (Part 1)": {
        "url": "https://archive.org/details/gov.in.is.9873.1.2012",
        "title": "BIS Official Standard - IS 9873 Part 1 (Safety of Toys)",
    },
    "IS 9873": {
        "url": "https://archive.org/details/gov.in.is.9873.1.2012",
        "title": "BIS Official Standard - IS 9873 (Safety of Toys)",
    },
    "IS 15644": {
        "url": "https://archive.org/details/gov.in.is.15644.2006",
        "title": "BIS Official Standard - IS 15644 (Electric Toys Safety)",
    },
    "IS 15636": {
        "url": "https://archive.org/details/gov.in.is.15636.2012",
        "title": "BIS Official Standard - IS 15636 (Passenger Car Pneumatic Tyres)",
    },

    # Hallmarking
    "IS 1417 (Gold Hallmarking)": {
        "url": "https://archive.org/details/gov.in.is.1417.2016",
        "title": "BIS Official Standard - IS 1417 : 2016 (Gold Hallmarking & Fineness)",
    },
    "IS 1417": {
        "url": "https://archive.org/details/gov.in.is.1417.2016",
        "title": "BIS Official Standard - IS 1417 : 2016 (Gold Hallmarking & Fineness)",
    },
    "IS 2112": {
        "url": "https://archive.org/details/gov.in.is.2112.2014",
        "title": "BIS Official Standard - IS 2112 : 2014 (Silver Hallmarking)",
    },
    "IS 2112 (Silver Hallmarking)": {
        "url": "https://archive.org/details/gov.in.is.2112.2014",
        "title": "BIS Official Standard - IS 2112 : 2014 (Silver Hallmarking)",
    },
}


def resolve_standard_urls(label_or_number):
    """
    Returns (doc_url, portal_url, source_title)
    doc_url: Direct link to official standard document reader/scanned PDF on Archive.org.
    portal_url: Official BIS Connect Know Your Standards lookup portal.
    """
    if not label_or_number:
        return (
            DEFAULT_PORTAL_URL,
            DEFAULT_PORTAL_URL,
            "BIS Standards Portal",
        )

    lbl = str(label_or_number).strip()

    # 1. Exact match in catalog
    if lbl in STANDARDS_DOC_MAP:
        info = STANDARDS_DOC_MAP[lbl]
        return (info["url"], DEFAULT_PORTAL_URL, info["title"])

    # 2. Extract standard number and part if present
    m = re.search(
        r"IS[\s:\-]?([0-9]{2,5})(?:\s*(?:\(Part\s*(\d+)\)|Part\s*(\d+)))?",
        lbl,
        re.IGNORECASE,
    )
    if m:
        num = m.group(1)
        part = m.group(2) or m.group(3)
        if part:
            candidate_part = f"IS {num} (Part {part})"
            if candidate_part in STANDARDS_DOC_MAP:
                info = STANDARDS_DOC_MAP[candidate_part]
                return (info["url"], DEFAULT_PORTAL_URL, info["title"])

        candidate_base = f"IS {num}"
        if candidate_base in STANDARDS_DOC_MAP:
            info = STANDARDS_DOC_MAP[candidate_base]
            return (info["url"], DEFAULT_PORTAL_URL, info["title"])

        # Fallback to authentic public safety standard search on Archive.org
        search_query = urllib.parse.quote(f'identifier:gov.in.is.{num}* OR title:"IS {num}"')
        doc_url = f"https://archive.org/search?query={search_query}"
        return (
            doc_url,
            DEFAULT_PORTAL_URL,
            f"BIS Official Standard - IS {num} Document",
        )

    return (
        DEFAULT_PORTAL_URL,
        DEFAULT_PORTAL_URL,
        f"BIS Standards Portal - {lbl}",
    )


def build_citations(hits):
    cites = []
    seen = set()
    for h in hits:
        e = h["entry"]
        cid = e.get("id") or e.get("is_number") or e.get("title")
        if cid in seen:
            continue
        seen.add(cid)

        is_num = e.get("is_number")
        doc_url, portal_url, fallback_title = resolve_standard_urls(is_num or e.get("title"))

        source_url = e.get("source_url")
        if not source_url or "standard_review" in source_url:
            source_url = doc_url or portal_url

        cites.append(
            {
                "id": e["id"],
                "label": is_num or e["title"],
                "source_title": e.get("source_title") or fallback_title,
                "source_url": source_url,
                "doc_url": e.get("doc_url") or doc_url,
                "portal_url": e.get("portal_url") or portal_url,
                "verified": e.get("verified", False),
                "confidence": round(float(h.get("score", 0.0)), 2),
            }
        )
    return cites


def extract_dynamic_citations(text, existing_citations):
    """
    Extracts Indian Standard numbers mentioned in response (e.g. IS 14543, IS 456, IS 15683)
    and constructs direct links to the official standard document and BIS portal.
    """
    existing_labels = {c["label"].upper() for c in existing_citations}
    new_cites = []

    pattern = re.compile(
        r"\bIS[\s:\-]?([0-9]{2,5}(?:\s*(?:\(Part\s*\d+\)|Part\s*\d+))?)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    seen_standards = set()

    for m in matches:
        clean_num = re.sub(r"\s+", " ", m.strip())
        std_label = f"IS {clean_num}"
        if std_label.upper() in existing_labels or std_label.upper() in seen_standards:
            continue
        seen_standards.add(std_label.upper())

        std_clean = clean_num.split()[0]
        doc_url, portal_url, source_title = resolve_standard_urls(std_label)

        new_cites.append(
            {
                "id": f"DYN-{std_clean}",
                "label": std_label,
                "source_title": source_title,
                "source_url": doc_url,
                "doc_url": doc_url,
                "portal_url": portal_url,
                "verified": True,
                "confidence": 9.5,
                "dynamic": True,
            }
        )

    return existing_citations + new_cites[:4]


def extract_standard_card(text, hits=None, question="", lang="en"):
    """
    Extracts a structured BIS Standard Card from either tagged AI output,
    local knowledge base entries, or regex analysis of the response.
    Supports authentic bilingual rendering (Hindi & English) and offline coverage.
    """
    is_hi = (lang == "hi")

    def attach_urls(cd):
        if not cd:
            return None
        lbl = cd.get("is_number") or cd.get("title")
        doc_u, port_u, _ = resolve_standard_urls(lbl)
        cd["doc_url"] = doc_u
        cd["portal_url"] = port_u
        return cd

    # 1. Try parsing explicit [STANDARD_CARD] tag block
    tag_match = re.search(r"\[STANDARD_CARD\](.*?)\[END_STANDARD_CARD\]", text, re.DOTALL)
    if tag_match:
        block = tag_match.group(1)

        def get_val(key):
            m = re.search(rf"{key}:\s*(.*?)(?=\n[A-Z_]+:|\Z)", block, re.DOTALL)
            return m.group(1).strip() if m else ""

        is_num = get_val("IS_NUMBER") or ("भारतीय मानक" if is_hi else "Indian Standard")
        title = get_val("TITLE")
        definition = get_val("DEFINITION")
        given_to = [x.strip() for x in get_val("GIVEN_TO").split(",") if x.strip()]
        not_given_to = [x.strip() for x in get_val("NOT_GIVEN_TO").split(",") if x.strip()]
        scheme = get_val("SCHEME") or ("योजना-I (ISI मार्क)" if is_hi else "Scheme-I (ISI Mark)")
        status = get_val("STATUS") or ("अनिवार्य QCO के अंतर्गत" if is_hi else "Mandatory under QCO")

        return attach_urls({
            "is_number": is_num,
            "title": title,
            "definition": definition,
            "given_to": given_to,
            "not_given_to": not_given_to,
            "scheme": scheme,
            "status": status,
        })

    # 2. If hits from local database are present (Offline Database Mode)
    if hits:
        top = hits[0]["entry"]
        is_num = top.get("is_number") or ""

        # Deduce standard/procedure identifier if missing
        if not is_num:
            topic = top.get("topic", "")
            title_lower = top.get("title", "").lower()
            id_lower = top.get("id", "").lower()
            if "hallmarking" in topic or "hallmark" in title_lower:
                is_num = "IS 1417 (बीआईएस हॉलमार्क एवं HUID)" if is_hi else "IS 1417 (BIS Hallmark & HUID)"
            elif "fee" in id_lower or "fee" in title_lower:
                is_num = "बीआईएस शुल्क संरचना" if is_hi else "BIS Fee Structure & MSME"
            elif "renewal" in id_lower or "renewal" in title_lower:
                is_num = "बीआईएस लाइसेंस नवीनीकरण" if is_hi else "BIS Licence Renewal (CM/L)"
            elif "audit" in id_lower or "audit" in title_lower:
                is_num = "कारखाना ऑडिट एवं निरीक्षण" if is_hi else "BIS Factory Audit (SIT)"
            elif "penalt" in id_lower or "penalt" in title_lower:
                is_num = "बीआईएस अधिनियम धारा 29" if is_hi else "BIS Act 2016 (Section 29)"
            elif "scheme-x" in id_lower or "scheme x" in title_lower:
                is_num = "योजना-X (मशीनरी एवं पूंजीगत सामान)" if is_hi else "Scheme-X (Machinery)"
            elif "eco" in id_lower or "eco" in title_lower:
                is_num = "इको मार्क पर्यावरण योजना" if is_hi else "ECO Mark Scheme"
            elif "crs" in id_lower or "crs" in title_lower:
                is_num = "योजना-II (अनिवार्य पंजीकरण CRS)" if is_hi else "Scheme-II (CRS)"
            elif "complaint" in id_lower or "consumer" in topic:
                is_num = "बीआईएस केयर ऐप एवं शिकायत" if is_hi else "BIS Care App & Verification"
            else:
                is_num = "आधिकारिक भारतीय मानक" if is_hi else "Indian Standard Specification"

        if is_hi:
            title = top.get("title_hi") or top.get("title", is_num)
            definition = top.get("summary_hi") or top.get("summary", "")
            hi_keywords = [k for k in top.get("keywords", []) if DEVANAGARI.search(k)]
            en_keywords = [k for k in top.get("keywords", []) if not DEVANAGARI.search(k)]
            given_to = (hi_keywords[:4] if hi_keywords else en_keywords[:4]) or ["मानक अनुरूप उत्पाद", "प्रमाणित निर्माता"]
            not_given_to = ["बिना मानक चिह्न वाले डुप्लीकेट सामान", "गैर-अनुपालन उत्पाद"]
            scheme = "योजना-I (ISI मार्क)" if top.get("topic") == "standard" else "बीआईएस प्रमाणन योजना"
            status = "अनिवार्य गुणवत्ता नियंत्रण आदेश (QCO)" if ("mandatory" in top.get("summary", "").lower() or "अनिवार्य" in str(top.get("summary_hi", ""))) else "आधिकारिक भारतीय मानक"
        else:
            title = top.get("title", is_num)
            definition = top.get("summary", "")
            given_to = [k for k in top.get("keywords", []) if not any(w in k.lower() for w in ["standard", "bis", "मानक"]) and not DEVANAGARI.search(k)][:4]
            if not given_to:
                given_to = ["Products conforming to standard specifications", "Certified manufacturers"]
            not_given_to = ["Non-specified variants", "Uncertified counterfeit goods"]
            scheme = "Scheme-I (ISI Mark)" if top.get("topic") == "standard" else "BIS Compliance Scheme"
            status = "Mandatory under QCO" if "mandatory" in top.get("summary", "").lower() else "Official Indian Standard"

        return attach_urls({
            "is_number": is_num,
            "title": title,
            "definition": definition,
            "given_to": given_to,
            "not_given_to": not_given_to,
            "scheme": scheme,
            "status": status,
        })

    # 3. Fallback heuristic from text
    is_matches = re.findall(
        r"\bIS[\s:\-]?([0-9]{2,5}(?:\s*(?:\(Part\s*\d+\)|Part\s*\d+))?)\b",
        text,
        re.IGNORECASE,
    )
    if is_matches:
        first_std = f"IS {is_matches[0].strip()}"
        first_para = (text.split("\n\n")[0] if "\n\n" in text else text).strip()
        return attach_urls({
            "is_number": first_std,
            "title": (f"भारतीय मानक {first_std}" if is_hi else f"Indian Standard {first_std}"),
            "definition": first_para[:220] + ("..." if len(first_para) > 220 else ""),
            "given_to": (["मानक अनुरूप विनिर्देश उत्पाद", "प्रमाणित निर्माता"] if is_hi else ["Products complying with standard specifications", "Certified manufacturers"]),
            "not_given_to": (["अमानक अनधिकृत वस्तुएं", "गैर-अनुपालक उत्पाद"] if is_hi else ["General uncertified articles", "Non-compliant items"]),
            "scheme": ("योजना-I (ISI मार्क)" if is_hi else "Scheme-I (ISI Mark)") if any(k in first_std for k in ["15298", "15683", "4151", "1786", "1489", "12269", "3196"]) else ("बीआईएस योजना" if is_hi else "BIS Scheme"),
            "status": ("आधिकारिक विनिर्देश" if is_hi else "Official Specification"),
        })

    # 4. If query is about non-mandatory or general item (like general chappals)
    q_lower = question.lower()
    if any(term in q_lower for term in ["chappal", "slipper", "sandal", "casual", "चप्पल"]):
        return attach_urls({
            "is_number": "ऐच्छिक / गैर-अनिवार्य" if is_hi else "Non-Mandatory / Voluntary",
            "title": "सामान्य फुटवियर एवं घरेलू चप्पल" if is_hi else "General Footwear & Casual Slippers",
            "definition": (
                "सामान्य कैजुअल चप्पल और घरेलू स्लीपर्स अनिवार्य बीआईएस प्रमाणन के अंतर्गत नहीं आते हैं। सुरक्षा जूते विनिर्देश (IS 15298) केवल औद्योगिक सुरक्षा जूतों पर लागू होते हैं।"
                if is_hi
                else "Standard casual chappals and slippers do not fall under compulsory BIS certification. Safety footwear specifications (IS 15298) apply only to industrial protective footwear with toe caps."
            ),
            "given_to": ["सामान्य उपभोक्ता फुटवियर", "स्वैच्छिक गुणवत्ता सत्यापन"] if is_hi else ["General Consumer Footwear", "Voluntary Quality Verification"],
            "not_given_to": ["औद्योगिक सुरक्षा जूते (इसके लिए IS 15298 Part 2 अनिवार्य है)"] if is_hi else ["Industrial Safety Footwear (requires IS 15298 Part 2)"],
            "scheme": "स्वैच्छिक (सामान्य चप्पल के लिए ISI मार्क अनिवार्य नहीं)" if is_hi else "Voluntary (No Mandatory ISI Mark required for standard chappals)",
            "status": "दैनिक उपयोग के लिए गैर-अनिवार्य" if is_hi else "Non-Compulsory for Casual Wear",
        })

    return None


def clean_card_tags(text):
    return re.sub(r"\[STANDARD_CARD\].*?\[END_STANDARD_CARD\]\s*", "", text, flags=re.DOTALL).strip()


def grounded_answer(hits, lang="en"):
    top = hits[0]["entry"]
    if lang == "hi" and top.get("summary_hi"):
        body = top["summary_hi"]
    else:
        body = top["summary"]

    parts = [body]

    labels = {
        "en": ("\nWhat to do next:", "\nAlso relevant: "),
        "hi": ("\nआगे क्या करें:", "\nये भी देखें: "),
    }
    steps_label, related_label = labels.get(lang, labels["en"])

    if lang == "hi" and top.get("next_steps_hi"):
        steps = top.get("next_steps_hi")
    else:
        steps = top.get("next_steps") or []

    if steps:
        parts.append(steps_label)
        for i, s in enumerate(steps, 1):
            parts.append(f"{i}. {s}")

    related = [h["entry"] for h in hits[1:]]
    if related:
        names = ", ".join(r.get("is_number") or r["title"] for r in related)
        parts.append(related_label + names)

    return "\n".join(parts)


def generate_follow_up_suggestions(card, question, lang="en"):
    """
    Generates dynamic, contextually relevant follow-up query suggestions
    based on the retrieved standard, scheme, and user inquiry.
    """
    std = (card.get("standard") or "").strip() if card else ""
    q_lower = (question or "").lower()

    if lang == "hi":
        if "1417" in std or "2112" in std or "हॉलमार्क" in q_lower or "hallmark" in q_lower:
            return [
                "🔍 बीआईएस केयर ऐप पर HUID कोड कैसे चेक करें?",
                "🏪 ज्वैलर मानकॉन्लाइन (Manakonline) पर पंजीकरण कैसे करें?",
                "⚖️ 2 ग्राम से कम वजन के आभूषणों पर क्या छूट है?",
                "📋 हॉलमार्किंग के लिए एएचसी (AHC) केंद्र में कौन से परीक्षण होते हैं?",
            ]
        elif std and std != "Non-Mandatory / Voluntary":
            return [
                f"💰 {std} के लिए आवेदन, परीक्षण और मार्किंग शुल्क कितना है?",
                f"📋 {std} के लिए फैक्ट्री ऑडिट और आवश्यक दस्तावेजों की सूची क्या है?",
                f"⚖️ क्या {std} के तहत गुणवत्ता नियंत्रण आदेश (QCO) अनिवार्य है?",
                f"🔄 {std} लाइसेंस की वैधता और नवीनीकरण प्रक्रिया क्या है?",
            ]
        elif "isi" in q_lower or "crs" in q_lower or "योजना" in q_lower or "scheme" in q_lower:
            return [
                "🏢 छोटे व्यवसाय (MSME) बीआईएस प्रमाणन प्रक्रिया कैसे शुरू करें?",
                "🔄 लाइसेंस समाप्त होने से पहले नवीनीकरण के क्या नियम हैं?",
                "⚖️ बीआईएस अधिनियम धारा 29 के तहत फर्जी मार्क पर क्या सजा है?",
                "🌱 पर्यावरण अनुकूल उत्पादों के लिए इको मार्क (ECO Mark) क्या है?",
            ]
        else:
            return [
                "💰 बीआईएस प्रमाणन शुल्क संरचना और एमएसएमई छूट क्या है?",
                "📱 बीआईएस केयर ऐप पर उत्पाद लाइसेंस कैसे सत्यापित करें?",
                "🏭 फैक्ट्री निरीक्षण और ऑडिट के दौरान क्या जांचा जाता है?",
                "📋 मानकॉन्लाइन पोर्टल पर ऑनलाइन आवेदन के चरण क्या हैं?",
            ]
    else:
        if "1417" in std or "2112" in std or "hallmark" in q_lower or "jewel" in q_lower or "gold" in q_lower or "silver" in q_lower:
            return [
                "🔍 How to verify the 6-digit HUID code on BIS Care App?",
                "🏪 How does a jeweller register for hallmarking on Manakonline?",
                "⚖️ What are the mandatory hallmarking exemptions under 2 grams?",
                "📋 What tests are conducted at an Assaying & Hallmarking Centre (AHC)?",
            ]
        elif std and std != "Non-Mandatory / Voluntary":
            return [
                f"💰 What are the application, testing & marking fees for {std}?",
                f"📋 What documents and factory audit steps are required for {std}?",
                f"⚖️ Is {std} mandatory under a Quality Control Order (QCO)?",
                f"🔄 What is the renewal procedure before {std} licence expiry?",
            ]
        elif "isi" in q_lower or "crs" in q_lower or "scheme" in q_lower or "msme" in q_lower:
            return [
                "🏢 How should an MSME or startup begin the BIS certification process?",
                "🔄 What is the renewal procedure and grace period for licences?",
                "⚖️ What are the penalties under Section 29 of the BIS Act for misuse?",
                "🌱 What is Scheme-X and how does it apply to machinery?",
            ]
        else:
            return [
                "💰 What is the fee structure and MSME concession for BIS certification?",
                "📱 How to verify licence authenticity on the BIS Care App?",
                "🏭 What happens during a BIS factory audit and inspection?",
                "📋 What are the step-by-step procedures on Manakonline?",
            ]


try:
    from agents_catalog import get_agent_by_id
except ImportError:
    try:
        from backend.agents_catalog import get_agent_by_id
    except ImportError:
        get_agent_by_id = lambda x: None


def call_groq(question, hits, lang="en", api_key=None, model=None, chat_history=None, agent_id=None):
    """
    Calls AI engine with context grounding, specialized BIS system prompt,
    specialized agent persona injection, multi-turn chat history, and automatic multi-key failover.
    Returns (answer_text, is_refusal, key_info) or raises an exception.
    """
    all_keys = get_all_groq_keys(api_key)
    if not all_keys:
        raise ValueError("No Groq API key configured on client or server")

    target_model = model or get_groq_model()

    context_blocks = []
    for h in hits:
        e = h["entry"]
        block = {
            "id": e.get("id"),
            "is_number": e.get("is_number"),
            "title": e.get("title"),
            "summary": e.get("summary_hi") if (lang == "hi" and e.get("summary_hi")) else e.get("summary"),
            "next_steps": (e.get("next_steps_hi") if (lang == "hi" and e.get("next_steps_hi")) else e.get("next_steps", [])),
            "source_url": e.get("source_url"),
        }
        context_blocks.append(json.dumps(block, ensure_ascii=False))

    context_str = "\n\n".join(context_blocks) if context_blocks else "No local seed matches found."

    agent = get_agent_by_id(agent_id) if agent_id else None
    agent_header = ""
    if agent:
        agent_header = (
            f"\n\nSPECIALIZED AGENT PERSONA ACTIVATED:\n"
            f"You are the {agent['name']} ({agent.get('scheme', 'BIS Compliance')}).\n"
            f"Mandate & Role: {agent.get('system_role', '')}\n"
            f"Primary Relevant Standards: {', '.join(agent.get('standards', []))}.\n"
            f"Key Diagnostic & Testing Directives: {agent.get('key_tests', '')}.\n"
        )

    system_prompt = (
        "You are BIS Sahayak (बीआईएस सहायक), the official conversational AI assistant for Indian Standards "
        "and Bureau of Indian Standards (BIS) services, under the Department of Consumer Affairs, Government of India "
        "(Smart India Hackathon Problem Statement 26107)."
        + agent_header + "\n\n"
        "YOUR OBJECTIVES:\n"
        "1. Give authoritative, accurate, and practical guidance on Indian Standards (IS), BIS certification, product quality, "
        "and consumer verification.\n"
        "2. CONVERSATIONAL MEMORY: If the user asks a follow-up query using pronouns or relative phrases (e.g. 'What are the fees for this?', "
        "'How do I apply for it?', 'Is it mandatory?', 'What documents are needed?'), resolve 'this/it' using the ongoing conversation history "
        "and answer authoritatively for the product/standard currently under discussion.\n"
        "3. SPECIFICITY: Always cite the exact Indian Standard number where applicable (e.g. IS 456 for Concrete, IS 10500 for Drinking Water, "
        "IS 4151 for Helmets, IS 14543 for Packaged Water, IS 694 for Cables, IS 1786 for TMT Rebars, IS 15298 for Safety Shoes, "
        "IS 1489 for PPC Cement, IS 12269 for OPC 53 Cement, IS 3196 for LPG Cylinders, IS 9873 for Toys, "
        "IS 15683 for Fire Extinguishers, IS 1417 for Gold Hallmarking, IS 2112 for Silver Hallmarking, IS 302 for Electrical Appliances, etc.).\n"
        "4. SCHEME ACCURACY: Clearly distinguish Scheme-I (ISI Mark with CM/L licence & factory audit), Scheme-II (CRS - Compulsory Registration "
        "for electronics/IT with R-number based on lab test reports), Scheme-IV (Hallmarking with 6-digit HUID), Scheme-X (low-risk machinery), and FMCS.\n"
        "5. CONSUMER TOOLS: Mention the official BIS Care Mobile App for verifying licences/HUID and filing complaints.\n"
        "6. NEXT STEPS: When relevant, outline practical next steps for the user (e.g. testing in BIS recognized labs, applying on manakonline.in).\n"
        "7. STRUCTURED OUTPUT: Begin every relevant response with this exact structured block:\n"
        "[STANDARD_CARD]\n"
        "IS_NUMBER: <Exact IS number e.g. 'IS 15298 (Part 2)', 'IS 15683', 'IS 1786', or 'Non-Mandatory / Voluntary' if not compulsory>\n"
        "TITLE: <Formal standard title>\n"
        "DEFINITION: <1-2 clear sentences explaining what this standard covers and its purpose>\n"
        "GIVEN_TO: <Comma-separated list of items this standard is given to / applies to>\n"
        "NOT_GIVEN_TO: <Comma-separated list of items/exceptions not covered (e.g. casual chappals, non-industrial items)>\n"
        "SCHEME: <Scheme-I (ISI Mark) | Scheme-II (CRS) | Scheme-IV (Hallmarking) | Scheme-X | Voluntary>\n"
        "STATUS: <Mandatory under QCO | Voluntary / Non-Compulsory>\n"
        "[END_STANDARD_CARD]\n\n"
        "8. CONCISENESS & COMPLETENESS: Be concise, direct, structured, and punchy. Avoid repetitive filler. "
        "ALWAYS fully conclude and complete every sentence, list item, step, and table. NEVER stop abruptly mid-sentence.\n\n"
        f"9. LANGUAGE: {'Respond in clear, professional Hindi (हिंदी).' if lang == 'hi' else 'Respond in clear English.'}\n\n"
        "STRICT REFUSAL RULE FOR UNRELATED QUERIES:\n"
        "If the user asks about topics completely unrelated to Indian standards, product manufacturing/safety, BIS certification, "
        "testing, or consumer quality (e.g. sports scores, entertainment/movies, restaurant/cooking recipes, weather, general investment/crypto, "
        "or military defense specs with no civilian BIS standard such as aircraft landing gear or submarine hull welding), "
        "you MUST start your response with '[REFUSAL]' and politely explain that BIS Sahayak is specialized exclusively in "
        "Indian Standards and Bureau of Indian Standards services.\n"
    )

    user_prompt = (
        f"Verified Seed Knowledge Base Context:\n{context_str}\n\n"
        f"User Question: {question}\n\n"
        "Provide a concise, complete, and authoritative BIS response with [STANDARD_CARD] at the top:"
    )

    candidate_models = [target_model] + [m for m in GROQ_FALLBACK_MODELS if m != target_model]
    key_errors = []

    # Iterate through each available Groq API key in the pool
    for key_idx, key in enumerate(all_keys, 1):
        key_masked = f"...{key[-6:]}" if len(key) >= 6 else "***"

        for m in candidate_models:
            # Multi-turn conversational message construction
            messages = [{"role": "system", "content": system_prompt}]

            if chat_history and isinstance(chat_history, list):
                for turn in chat_history[-6:]:
                    r = turn.get("role")
                    c = (turn.get("content") or "").strip()
                    if r in ("user", "assistant") and c:
                        if r == "assistant" and "[STANDARD_CARD]" in c:
                            c = re.sub(r"\[STANDARD_CARD\].*?\[END_STANDARD_CARD\]", "", c, flags=re.DOTALL).strip()
                        if len(c) > 600:
                            c = c[:600] + "..."
                        messages.append({"role": r, "content": c})

            messages.append({"role": "user", "content": user_prompt})

            payload = {
                "model": m,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 2500,
            }

            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                GROQ_API_URL,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_body = resp.read().decode("utf-8")
                    data = json.loads(res_body)

                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                message_content = (msg.get("content") or "").strip()
                if not message_content and msg.get("reasoning"):
                    message_content = msg.get("reasoning", "").strip()

                if not message_content:
                    continue

                is_refusal = any(term in message_content for term in REFUSAL_TERMS)
                clean_content = message_content.replace("[REFUSAL]", "").strip() if is_refusal else message_content
                key_info = {
                    "key_index": key_idx,
                    "total_keys": len(all_keys),
                    "failovers": key_idx - 1,
                    "model": m,
                }
                return clean_content, is_refusal, key_info

            except urllib.error.HTTPError as err:
                # 404 means model not found on Groq, try next candidate model with same key
                if err.code == 404:
                    continue

                # 429 (Rate limit) or 401/403 (Invalid/Revoked key)
                err_msg = f"HTTP {err.code}: {err.reason}"
                import sys
                sys.stderr.write(
                    f"[Groq Key Failover] Key #{key_idx} ({key_masked}) failed with {err_msg}. "
                    f"{'Switching to next key...' if key_idx < len(all_keys) else 'All keys exhausted.'}\n"
                )
                key_errors.append(f"Key #{key_idx} ({key_masked}): {err_msg}")
                break  # Stop trying models for this failed key; try next key

            except Exception as err:
                import sys
                sys.stderr.write(
                    f"[Groq Key Failover] Key #{key_idx} ({key_masked}) error: {err}. "
                    f"{'Switching to next key...' if key_idx < len(all_keys) else 'All keys exhausted.'}\n"
                )
                key_errors.append(f"Key #{key_idx} ({key_masked}): {str(err)}")
                break

    raise RuntimeError(
        f"All {len(all_keys)} Groq API key(s) failed. Errors: {'; '.join(key_errors)}"
    )


def compose(question, hits, lang="en", mode="auto", api_key=None, model=None, chat_history=None, agent_id=None):
    """
    Main answering orchestrator with dynamic AI, multi-key failover,
    specialized category agent persona, multi-turn conversational context,
    and automatic offline database fallback.
    """
    # 1. Dataset-only mode explicitly requested
    if mode in ("dataset", "local", "grounded"):
        if not hits:
            return {
                "answer": NO_ANSWER.get(lang, NO_ANSWER["en"]),
                "citations": [],
                "standard_card": None,
                "mode": "no_match",
                "engine": "BIS Verified Database (Local)",
                "fallback": False,
                "agent_id": agent_id,
                "follow_up_suggestions": generate_follow_up_suggestions(None, question, lang=lang),
            }
        ans = grounded_answer(hits, lang)
        card = extract_standard_card(ans, hits, question, lang=lang)
        return {
            "answer": ans,
            "citations": build_citations(hits),
            "standard_card": card,
            "mode": "grounded",
            "engine": "BIS Verified Database (Local)",
            "fallback": False,
            "agent_id": agent_id,
            "follow_up_suggestions": generate_follow_up_suggestions(card, question, lang=lang),
        }

    # 2. Dynamic AI mode (auto or groq) with Multi-Key Failover
    all_keys = get_all_groq_keys(api_key)

    if all_keys:
        try:
            ai_text, is_refusal, key_info = call_groq(
                question, hits, lang=lang, api_key=api_key, model=model, chat_history=chat_history, agent_id=agent_id
            )

            engine_label = "BIS Sahayak AI"
            if key_info["failovers"] > 0:
                engine_label += f" (Key #{key_info['key_index']}/{key_info['total_keys']} - Failover Active)"
            elif key_info["total_keys"] > 1:
                engine_label += f" (Pool: {key_info['total_keys']} Keys Armed)"

            if is_refusal:
                return {
                    "answer": clean_card_tags(ai_text),
                    "citations": [],
                    "standard_card": None,
                    "mode": "no_match",
                    "engine": engine_label,
                    "dynamic": True,
                    "fallback": False,
                    "key_info": key_info,
                    "follow_up_suggestions": generate_follow_up_suggestions(None, question, lang=lang),
                }

            initial_citations = build_citations(hits)
            all_citations = extract_dynamic_citations(ai_text, initial_citations)
            card = extract_standard_card(ai_text, hits, question, lang=lang)
            cleaned_text = clean_card_tags(ai_text)
            follow_ups = generate_follow_up_suggestions(card, question, lang=lang)

            return {
                "answer": cleaned_text,
                "citations": all_citations,
                "standard_card": card,
                "mode": "groq",
                "engine": engine_label,
                "dynamic": True,
                "fallback": False,
                "key_info": key_info,
                "agent_id": agent_id,
                "follow_up_suggestions": follow_ups,
            }
        except Exception as err:
            if mode in ("groq", "grok"):
                return {
                    "answer": f"AI Engine Error: {str(err)}. Please check your API keys or network connection.",
                    "citations": build_citations(hits),
                    "standard_card": None,
                    "mode": "error",
                    "engine": "BIS Sahayak AI (Error)",
                    "dynamic": False,
                    "fallback": False,
                    "agent_id": agent_id,
                    "follow_up_suggestions": generate_follow_up_suggestions(None, question, lang=lang),
                }
            fallback_reason = f"All {len(all_keys)} Groq API key(s) failed or rate-limited ({str(err)}). Switched to offline verified BIS database."
    else:
        fallback_reason = "No Groq API keys configured. Using offline verified BIS database."

    # 3. Fallback to local dataset (when mode == "auto" and AI was unavailable or failed)
    if hits:
        ans = grounded_answer(hits, lang)
        card = extract_standard_card(ans, hits, question, lang=lang)
        return {
            "answer": ans,
            "citations": build_citations(hits),
            "standard_card": card,
            "mode": "grounded",
            "engine": "BIS Verified Database (Offline Fallback)",
            "dynamic": False,
            "fallback": True,
            "fallback_reason": fallback_reason,
            "agent_id": agent_id,
            "follow_up_suggestions": generate_follow_up_suggestions(card, question, lang=lang),
        }

    # If asking about an item not in seed, synthesize non-mandatory or scope card if appropriate
    fallback_card = extract_standard_card("", hits, question, lang=lang)
    return {
        "answer": NO_ANSWER.get(lang, NO_ANSWER["en"]),
        "citations": [],
        "standard_card": fallback_card,
        "mode": "no_match",
        "engine": "BIS Verified Database (Offline Fallback)",
        "dynamic": False,
        "fallback": True,
        "fallback_reason": fallback_reason,
        "agent_id": agent_id,
        "follow_up_suggestions": generate_follow_up_suggestions(fallback_card, question, lang=lang),
    }


INDIC_LANGUAGES = {
    "hi": ("Hindi", "हिन्दी"),
    "ta": ("Tamil", "தமிழ்"),
    "te": ("Telugu", "తెలుగు"),
    "bn": ("Bengali", "বাংলা"),
    "mr": ("Marathi", "मराठी"),
    "gu": ("Gujarati", "ગુજરાતી"),
    "kn": ("Kannada", "ಕನ್ನಡ"),
    "en": ("English", "English"),
}


def translate_text(text, target_lang="hi", api_key=None, model=None, standard_card=None):
    """
    Translates BIS answer and optional standard card into one of the Indian regional languages
    with multi-key failover support.
    """
    target_lang = (target_lang or "").lower().strip()
    if target_lang not in INDIC_LANGUAGES or target_lang == "en":
        return text, standard_card

    lang_name, native_name = INDIC_LANGUAGES[target_lang]

    all_keys = get_all_groq_keys(api_key)
    if not all_keys:
        return text, standard_card

    target_model = model or get_groq_model()

    system_prompt = (
        f"You are an expert official translator for the Government of India and the Bureau of Indian Standards (BIS).\n"
        f"Translate the following Indian Standards information into {lang_name} ({native_name}).\n"
        f"CRITICAL TRANSLATION RULES:\n"
        f"1. PRESERVE exact Indian Standard numbers (e.g. IS 1786, IS 456, IS 15298, IS 10500, IS 4151, IS 1417, IS 3196, IS 1489).\n"
        f"2. PRESERVE official scheme designations (Scheme-I, Scheme-II, Scheme-IV, Scheme-X, ISI Mark, CRS, CM/L, HUID, BIS Care, Manakonline).\n"
        f"3. PRESERVE all Markdown formatting: tables, bullet points, numbered lists, bolding, and headers.\n"
        f"4. If a [STANDARD_CARD] block is present, translate TITLE, DEFINITION, GIVEN_TO, and NOT_GIVEN_TO values into {lang_name}, while keeping the tag names and keys (IS_NUMBER, TITLE, DEFINITION, GIVEN_TO, NOT_GIVEN_TO, SCHEME, STATUS) in English.\n"
        f"5. Maintain an authoritative, professional, and clear public advisory tone.\n"
        f"6. Output ONLY the translated content without preamble or extra conversational remarks."
    )

    card_str = ""
    if standard_card:
        card_str = (
            f"[STANDARD_CARD]\n"
            f"IS_NUMBER: {standard_card.get('is_number', '')}\n"
            f"TITLE: {standard_card.get('title', '')}\n"
            f"DEFINITION: {standard_card.get('definition', '')}\n"
            f"GIVEN_TO: {', '.join(standard_card.get('given_to', []))}\n"
            f"NOT_GIVEN_TO: {', '.join(standard_card.get('not_given_to', []))}\n"
            f"SCHEME: {standard_card.get('scheme', '')}\n"
            f"STATUS: {standard_card.get('status', '')}\n"
            f"[END_STANDARD_CARD]\n\n"
        )

    full_content = card_str + text
    candidate_models = [target_model] + [m for m in GROQ_FALLBACK_MODELS if m != target_model]

    # Iterate through keys in pool for translation
    for key_idx, key in enumerate(all_keys, 1):
        for m in candidate_models:
            payload = {
                "model": m,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_content},
                ],
                "temperature": 0.1,
                "max_tokens": 2500,
            }

            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                GROQ_API_URL,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                res_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if not res_text:
                    continue

                trans_card = extract_standard_card(res_text) or standard_card
                trans_answer = clean_card_tags(res_text)
                return trans_answer, trans_card
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    continue
                # Rate limit (429) or auth error -> advance to next key in pool
                break
            except Exception:
                break

    return text, standard_card


def stream_groq(
    question,
    hits,
    lang="en",
    api_key=None,
    model=None,
    chat_history=None,
    agent_id=None,
    agent_persona=None
):
    """
    Generator that streams LLM tokens using Groq OpenAI-compatible SSE.
    Yields chunks of text as they arrive from Groq API.
    """
    all_keys = get_all_groq_keys(api_key)
    if not all_keys:
        # Fallback offline mode if no keys
        offline_resp = compose(question, hits, lang=lang, mode="dataset", agent_id=agent_id)
        yield {"token": offline_resp.get("answer", "")}
        return

    key = all_keys[0]
    target_model = model or DEFAULT_GROQ_MODEL
    
    agent_info = None
    if agent_id:
        from backend.agents_catalog import get_agent_by_id
        agent_info = get_agent_by_id(agent_id)

    system_prompt = build_system_prompt(hits, lang=lang, agent_persona=agent_info)
    user_prompt = build_user_prompt(question, lang=lang, agent_persona=agent_info)

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history and isinstance(chat_history, list):
        for turn in chat_history[-6:]:
            r = turn.get("role")
            c = (turn.get("content") or "").strip()
            if r in ("user", "assistant") and c:
                if r == "assistant" and "[STANDARD_CARD]" in c:
                    c = re.sub(r"\[STANDARD_CARD\].*?\[END_STANDARD_CARD\]", "", c, flags=re.DOTALL).strip()
                messages.append({"role": r, "content": c[:600]})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": target_model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500,
        "stream": True,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "BIS-Sahayak-Stream/1.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield {"token": token}
                    except Exception:
                        continue
    except Exception as e:
        import sys
        sys.stderr.write(f"[Stream Error] {e}\n")
        yield {"token": f"\n\n[Stream interrupted: {e}]"}



