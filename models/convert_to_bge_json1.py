import json

INPUT_FILE = "data/train_contract_index.json"
OUTPUT_FILE = "data/train_bge.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    contracts = json.load(f)

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for contract in contracts:

        sentences = contract["sentences"]

        for qa in contract["questions"]:

            positives = []

            for sid in qa["positive_sentence_ids"]:

                if sid < len(sentences):

                    positives.append(
                        sentences[sid]["text"].strip()
                    )

            if len(positives) == 0:
                continue

            sample = {

                "query": qa["question"].strip(),

                "pos": positives,

                "neg": []

            }

            out.write(
                json.dumps(sample, ensure_ascii=False)
            )

            out.write("\n")

            count += 1

print(f"Contracts : {len(contracts)}")
print(f"Samples   : {count}")
print(f"Saved     : {OUTPUT_FILE}")

import json

counts = []

with open("data/train_bge_hn.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        counts.append(len(sample["neg"]))

print("Min:", min(counts))
print("Max:", max(counts))
print("Average:", sum(counts)/len(counts))