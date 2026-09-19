"""
Run before every push:  python3 tests/test_demo_queries.py

Checks that each demo query answers or refuses as intended, and that the
knowledge base file still parses. Exits non-zero on failure, so it can go in
a pre-push hook or CI later.

Add your own query to CASES whenever you add entries. If a query you expect to
work starts failing, that is the retriever telling you a keyword is missing.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app import handle_ask  # noqa: E402

# (query, expected, expected label fragment in the top citation or None)
CASES = [
    # Hanswarup - Certification, Process, Fees, Audit, Renewal, Penalties, Schemes
    ("I manufacture helmets in Agra. What do I need?", "answer", "4151"),
    ("What is the difference between ISI mark and CRS?", "answer", "ISI"),
    ("what are the fees for BIS certification for MSME", "answer", "Fee"),
    ("how do I renew BIS licence before expiry", "answer", "Renewal"),
    ("what happens during BIS factory audit and inspection", "answer", "Audit"),
    ("penalties under BIS Act section 29 for fake mark", "answer", "Penalt"),
    ("what is scheme x for machinery", "answer", "Scheme-X"),
    ("what is eco mark scheme for environment", "answer", "ECO"),
    ("how do I start BIS certification for my small business", "answer", None),

    # Araz - Steel, Cement, LPG, Food Container, Packaged Water, Toys, Appliances
    ("sariya", "answer", "1786"),
    ("TMT bar 8mm rod specification", "answer", "1786"),
    ("cement standard PPC flyash", "answer", "1489"),
    ("OPC 53 grade cement standard", "answer", "12269"),
    ("LPG gas cylinder certification", "answer", "3196"),
    ("plastic food container tiffin standard", "answer", "10146"),
    ("packaged drinking water bottle specification", "answer", "14543"),
    ("safety of toys ISI mark children", "answer", "9873"),
    ("LED bulb lamp standard", "answer", "16102"),
    ("domestic pressure cooker standard", "answer", "2347"),
    ("gas stove chulha standard", "answer", "4246"),
    ("Where can I get drinking water tested?", "answer", "10500"),
    ("what does IS 10500 cover", "answer", "10500"),
    ("I want to sell cables, what certification do I need", "answer", "694"),
    ("IS 456", "answer", "456"),

    # Shreyansh - Footwear and Leather Standards
    ("which standard for safety shoes 200 joules", "answer", "15298"),
    ("protective footwear 100 joules", "answer", "15298"),
    ("occupational footwear without toe cap", "answer", "15298"),
    ("pvc chappal sandal casual footwear", "answer", "6721"),
    ("sports running footwear standard", "answer", "15844"),
    ("leather school shoes derby specification", "answer", "11544"),

    # Yukti - Hallmarking and Consumer
    ("How do I check if my gold jewellery is really hallmarked?", "answer", "hallmark"),
    ("silver necklace hallmarking 925 sterling", "answer", "2112"),
    ("what standard applies to silver jewellery", "answer", "2112"),
    ("how do I register as a jeweller for hallmarking", "answer", "jeweller"),
    ("mandatory hallmarking exemptions 2 grams", "answer", "hallmark"),
    ("how to file complaint on BIS care app", "answer", None),

    # Mudit - Hindi Queries with Hindi summary and next steps
    ("सोने के आभूषण का हॉलमार्क कैसे जाँचें?", "answer", "hallmark"),
    ("चाँदी के आभूषणों और गहनों के हॉलमार्क का मानक?", "answer", "2112"),
    ("मैं हेलमेट बनाता हूँ, मुझे कौन सा प्रमाणन चाहिए?", "answer", "4151"),
    ("ISI चिह्न और CRS में क्या अंतर है?", "answer", None),
    ("पेयजल के लिए कौन सा मानक है?", "answer", "10500"),
    ("सरिया टीएमटी बार के लिए कौन सा मानक है?", "answer", "1786"),
    ("पीपीसी सीमेंट के लिए कौन सा मानक है?", "answer", "1489"),
    ("एलपीजी गैस सिलेंडर का मानक क्या है?", "answer", "3196"),
    ("बच्चों के खिलौने की सुरक्षा का मानक?", "answer", "9873"),
    ("सुरक्षा जूते सेफ्टी शू के लिए कौन सा मानक है?", "answer", "15298"),

    # Yukti - Refusal Path: 5 questions that MUST be refused in both languages
    ("Which standard applies to aircraft landing gear?", "refuse", None),
    ("What is the best pizza in Delhi?", "refuse", None),
    ("Which standard applies to submarine hull welding?", "refuse", None),
    ("Who won the cricket match yesterday?", "refuse", None),
    ("How do I invest in cryptocurrency or bitcoin?", "refuse", None),
    ("हवाई जहाज के लैंडिंग गियर के लिए कौन सा मानक है?", "refuse", None),
    ("दिल्ली में सबसे अच्छा पिज्जा कहाँ मिलता है?", "refuse", None),
    ("पनडुब्बी के पतवार वेल्डिंग के लिए कौन सा मानक है?", "refuse", None),
    ("कल का क्रिकेट मैच कौन जीता?", "refuse", None),
    ("क्रिप्टोकरेंसी या बिटकॉइन में निवेश कैसे करें?", "refuse", None),
]


def check_kb():
    path = os.path.join(ROOT, "data", "knowledge_base.json")
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)
    ids = [e["id"] for e in kb["entries"]]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate entry ids")
    for e in kb["entries"]:
        for field in ("id", "topic", "title", "summary", "summary_hi", "keywords", "next_steps", "next_steps_hi", "source_url"):
            if not e.get(field):
                problems.append(f"{e.get('id', '?')} missing {field}")
    unverified = sum(1 for e in kb["entries"] if not e.get("verified"))
    return len(kb["entries"]), unverified, problems


def main():
    total, unverified, problems = check_kb()
    print(f"Knowledge base: {total} entries, {unverified} not yet verified")
    for p in problems:
        print("  SCHEMA:", p)

    failures = []
    for query, expected, fragment in CASES:
        result, _ = handle_ask({"question": query, "mode": "dataset"})
        got = "refuse" if result["mode"] == "no_match" else "answer"
        ok = got == expected
        if ok and fragment and result["citations"]:
            ok = fragment.lower() in result["citations"][0]["label"].lower()
        if not ok:
            failures.append((query, expected, got, result["citations"][:1]))
        mark = "ok  " if ok else "FAIL"
        top = result["citations"][0]["label"][:34] if result["citations"] else "-"
        try:
            print(f"{mark} [{result['lang']}] {query[:48]:<50} {got:<7} {top}")
        except UnicodeEncodeError:
            safe_query = query[:48].encode("ascii", errors="replace").decode("ascii")
            safe_top = top.encode("ascii", errors="replace").decode("ascii")
            print(f"{mark} [{result['lang']}] {safe_query:<50} {got:<7} {safe_top}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} queries passed")
    if failures or problems:
        print("\nFix these before pushing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
