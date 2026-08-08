import json
from pathlib import Path

DATA_PATH = Path("data/train_separate_questions.json")


def load_cuad():
    """Load the CUAD training dataset."""

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":

    data = load_cuad()

    print("Version:", data["version"])

    print("\nNumber of contracts:")
    print(len(data["data"]))

    first_contract = data["data"][0]

    print("\nKeys in first contract:")
    print(first_contract.keys())

    print("\nTitle:")
    print(first_contract["title"])

    print("\nParagraph keys:")
    print(first_contract["paragraphs"][0].keys())