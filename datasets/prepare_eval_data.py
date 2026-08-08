import json
from collections import Counter

with open("data/test_contract_index.json","r",encoding="utf-8") as f:
    contracts = json.load(f)

counter = Counter()

for contract in contracts:
    for qa in contract["questions"]:
        counter[len(qa["positive_sentence_ids"])] += 1

print(counter)