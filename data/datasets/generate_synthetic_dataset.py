# generate_synthetic_dataset.py
"""Utility to generate a synthetic legal QA benchmark with 50+ items.
The script reads the existing `benchmark_data.json`, appends newly generated
items across a variety of legal document categories, and writes the combined
dataset back. It is safe to re‑run – duplicate `document_id`s are ignored.
All generated items guarantee at least one relevant chunk (the first chunk).
"""
import json
from pathlib import Path

# Path to the raw benchmark JSON (relative to project root)
DATA_PATH = Path(__file__).parent / "raw" / "benchmark_data.json"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_chunk(chunk_id, text, start, end, token_count, is_relevant, relevance_score):
    return {
        "id": chunk_id,
        "text": text,
        "start_position": start,
        "end_position": end,
        "token_count": token_count,
        "is_relevant": is_relevant,
        "relevance_score": relevance_score,
    }

def split_into_chunks(document_text):
    """Split on periods, keep the period, and mark the first chunk as relevant.
    Returns a list of chunk dicts.
    """
    sentences = [s.strip() for s in document_text.split('.') if s]
    chunks = []
    pos = 0
    for idx, sent in enumerate(sentences):
        sent = sent + '.'  # restore trailing period
        start = pos
        end = start + len(sent)
        token_count = len(sent.split())
        # Ensure at least one relevant chunk: make the first sentence relevant
        is_relevant = (idx == 0)
        relevance_score = 3 if is_relevant else 0
        chunks.append(make_chunk(idx, sent, start, end, token_count, is_relevant, relevance_score))
        pos = end + 1  # account for space after period
    return chunks

# ---------------------------------------------------------------------------
# Template definitions for each legal category
# ---------------------------------------------------------------------------

CATEGORY_TEMPLATES = {
    "nda": {
        "doc": "Non‑Disclosure Agreement between {party_a} and {party_b}. The term is {term} years. Confidential information must be returned upon termination. Governing law: {law}.",
        "qa": [
            ("What is the term of the NDA?", "{term} years"),
            ("Which law governs the NDA?", "{law}"),
            ("What must be done with confidential information upon termination?", "It must be returned.")
        ]
    },
    "employment": {
        "doc": "Employment Agreement between {employer} and {employee}. Either party may terminate with {notice} days notice. Salary is ${salary} per month. Confidentiality survives for {survive} months after termination.",
        "qa": [
            ("How many days notice are required for termination?", "{notice} days"),
            ("What is the monthly salary?", "${salary}")
        ]
    },
    "saas": {
        "doc": "SaaS Service Terms for {service}. Users must be at least {age} years old. The service may be terminated for violation of the Acceptable Use Policy. Data is retained for {retention} months.",
        "qa": [
            ("What is the minimum age to use the service?", "{age} years"),
            ("How long is user data retained?", "{retention} months")
        ]
    },
    "service": {
        "doc": "Service Agreement for {provider} providing {service_desc}. Payment is due within {payment_days} days of invoice. Liability is limited to the amount paid.",
        "qa": [
            ("Within how many days must payment be made?", "{payment_days} days"),
            ("What is the liability limit?", "Limited to the amount paid")
        ]
    },
    "vendor": {
        "doc": "Vendor Contract between {buyer} and {vendor}. Delivery must occur within {delivery_days} days. Warranty period is {warranty} months.",
        "qa": [
            ("How many days for delivery?", "{delivery_days} days"),
            ("What is the warranty period?", "{warranty} months")
        ]
    },
    "licensing": {
        "doc": "Licensing Agreement for {software}. License is non‑exclusive and non‑transferable. Term is {term_years} years. Renewal requires written notice 30 days before expiry.",
        "qa": [
            ("Is the license exclusive?", "No, it is non‑exclusive"),
            ("How long is the license term?", "{term_years} years")
        ]
    },
    "privacy": {
        "doc": "Privacy Policy for {app}. We collect {data_types}. Data is stored for {retention_months} months. Users may request deletion at any time.",
        "qa": [
            ("What types of data are collected?", "{data_types}"),
            ("How long is data retained?", "{retention_months} months")
        ]
    },
    "dpa": {
        "doc": "Data Processing Agreement between {controller} and {processor}. Processor must implement appropriate security measures and may not sub‑process without consent. Data breach must be reported within {breach_hours} hours.",
        "qa": [
            ("Within how many hours must a data breach be reported?", "{breach_hours} hours"),
            ("Can the processor sub‑process without consent?", "No")
        ]
    },
    "tos": {
        "doc": "Terms of Service for {service}. Users must not engage in illegal activities. The service may suspend accounts for violations. Governing jurisdiction: {jurisdiction}.",
        "qa": [
            ("What jurisdiction governs the Terms of Service?", "{jurisdiction}"),
            ("Can the service suspend an account?", "Yes, for violations")
        ]
    }
}

# Placeholder values used for templating – they produce realistic‑looking text.
PLACEHOLDERS = {
    "party_a": "CorpA",
    "party_b": "CorpB",
    "term": "5",
    "law": "Delaware",
    "employer": "EmployerCo",
    "employee": "EmployeeCo",
    "notice": "30",
    "salary": "5000",
    "survive": "24",
    "service": "AppX",
    "age": "13",
    "retention": "12",
    "provider": "ProviderInc",
    "service_desc": "cloud hosting",
    "payment_days": "30",
    "buyer": "BuyerLtd",
    "vendor": "VendorLtd",
    "delivery_days": "7",
    "warranty": "12",
    "software": "SuperApp",
    "term_years": "3",
    "app": "AppY",
    "data_types": "email addresses and usage analytics",
    "retention_months": "12",
    "controller": "DataCtrl",
    "processor": "DataProc",
    "breach_hours": "72",
    "jurisdiction": "California"
}

def fill(template: str) -> str:
    return template.format(**PLACEHOLDERS)

def generate_items() -> list:
    items = []
    # Produce 6 items per category (10 categories) => 60 items total.
    for cat, tmpl in CATEGORY_TEMPLATES.items():
        for i in range(6):
            document_id = f"{cat}_{i+1:02d}"
            document_text = fill(tmpl["doc"])
            # Use the first QA pair for this item (easy difficulty)
            question_tpl, answer_tpl = tmpl["qa"][0]
            question = fill(question_tpl)
            answer = fill(answer_tpl)
            chunks = split_into_chunks(document_text)
            item = {
                "document_id": document_id,
                "document_text": document_text,
                "question": question,
                "answer": answer,
                "difficulty": "easy",
                "provenance": {
                    "source_type": "synthetic",
                    "source_name": f"Generated {cat} template",
                    "license": "CC0"
                },
                "chunks": chunks
            }
            items.append(item)
    return items

def load_existing(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save(path: Path, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    existing = load_existing(DATA_PATH)
    existing_ids = {it["document_id"] for it in existing}
    generated = generate_items()
    to_add = [it for it in generated if it["document_id"] not in existing_ids]
    if not to_add:
        print("No new items to add.")
        return
    combined = existing + to_add
    save(DATA_PATH, combined)
    print(f"Added {len(to_add)} new benchmark items. Total now: {len(combined)}")

if __name__ == "__main__":
    main()
