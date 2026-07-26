# fix_benchmark_data.py
"""Utility to clean up benchmark_data.json after synthetic generation.
It recomputes token counts for each chunk, ensures each item has at least one
relevant chunk (marks the first chunk as relevant if none), and aligns
relevance_score with the is_relevant flag.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "raw" / "benchmark_data.json"

def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fix_item(item):
    # Recompute token counts for each chunk and ensure consistency
    any_relevant = False
    for chunk in item.get("chunks", []):
        text = chunk.get("text", "")
        # Recompute token count based on whitespace split
        chunk["token_count"] = len(text.split())
        # Ensure relevance_score aligns with is_relevant
        if chunk.get("is_relevant"):
            any_relevant = True
            # If relevance_score is 0, set to 3 (essential) as a safe default
            if chunk.get("relevance_score", 0) == 0:
                chunk["relevance_score"] = 3
        else:
            chunk["relevance_score"] = 0
    # If no relevant chunk, make the first one relevant
    if not any_relevant and item.get("chunks"):
        first = item["chunks"][0]
        first["is_relevant"] = True
        first["relevance_score"] = 3
    return item

def main():
    data = load_data()
    fixed = [fix_item(item) for item in data]
    save_data(fixed)
    print(f"Fixed {len(fixed)} items in benchmark_data.json")

if __name__ == "__main__":
    main()
