import json
from pathlib import Path

from sentence_splitter import split_sentences
from label_generator import find_positive_sentences

DATA_PATH = Path("data/train_separate_questions.json")
OUTPUT_PATH = Path("data/train_contract_index.json") 


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():

    data = load_data()

    contracts = []

    for contract_id, contract in enumerate(data["data"]):

        title = contract["title"]

        paragraph = contract["paragraphs"][0]

        context = paragraph["context"]

        sentences = split_sentences(context)

        contract_entry = {

            "contract_id": contract_id,

            "title": title,

            "sentences": sentences,

            "questions": []

        }

        for question_id, qa in enumerate(paragraph["qas"]):

            positive = list(
                find_positive_sentences(
                    sentences,
                    qa["answers"]
                )
            )

            contract_entry["questions"].append({

                "question_id": question_id,

                "question": qa["question"],

                "positive_sentence_ids": positive

            })

        contracts.append(contract_entry)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            contracts,
            f,
            indent=2
        )

    print()

    print("Contracts:", len(contracts))

    print("Saved to")

    print(OUTPUT_PATH)


if __name__ == "__main__":

    build_index()