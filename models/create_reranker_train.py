from FlagEmbedding import FlagModel
import json
import numpy as np
from tqdm import tqdm
import contextlib
import os

# =====================================================
# Configuration
# =====================================================

INPUT_FILE = "/kaggle/input/datasets/madhavkumar244/madhavscuadhn/train_contract_index.json"
OUTPUT_FILE = "/kaggle/working/train_bge_same_contract_hn.jsonl"

RETRIEVER_PATH = "Madhav2832005/bge-base-legal-retriever"

TOP_K = 20          # retrieve top-20
NEGATIVE_NUMBER = 15

BATCH_SIZE = 256
QUERY_BATCH_SIZE = 64
MAX_LENGTH = 384


# =====================================================
# Hide FlagEmbedding progress bars
# =====================================================

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield


# =====================================================
# Load retriever
# =====================================================

print("Loading retriever...")

retriever = FlagModel(
    RETRIEVER_PATH,
    devices="cuda:0",
    use_fp16=True
)

print("Retriever loaded.\n")


# =====================================================
# Load contracts
# =====================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    contracts = json.load(f)


total_samples = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for contract in tqdm(contracts):

        sentences = contract["sentences"]
        questions = contract["questions"]

        sentence_texts = [
            s["text"].strip()
            for s in sentences
        ]

        if len(sentence_texts) == 0:
            continue

        # --------------------------------------------
        # Encode contract sentences ONCE
        # --------------------------------------------

        with suppress_output():

            sentence_embeddings = retriever.encode_corpus(
                sentence_texts,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH
            )

        sentence_embeddings = np.asarray(sentence_embeddings)

        # --------------------------------------------
        # Collect valid questions
        # --------------------------------------------

        valid_questions = []
        positive_sets = []

        for qa in questions:

            positives = set(qa["positive_sentence_ids"])

            if len(positives) == 0:
                continue

            valid_questions.append(qa["question"].strip())
            positive_sets.append(positives)

        if len(valid_questions) == 0:
            continue

        # --------------------------------------------
        # Encode all questions together
        # --------------------------------------------

        with suppress_output():

            query_embeddings = retriever.encode_queries(
                valid_questions,
                batch_size=QUERY_BATCH_SIZE,
                max_length=MAX_LENGTH
            )

        query_embeddings = np.asarray(query_embeddings)

        # --------------------------------------------
        # Create training samples
        # --------------------------------------------

        for question, q_emb, positives in zip(
            valid_questions,
            query_embeddings,
            positive_sets
        ):

            scores = sentence_embeddings @ q_emb

            order = np.argsort(-scores)

            negatives = []

            for idx in order:

                if idx in positives:
                    continue

                negatives.append(sentence_texts[idx])

                if len(negatives) == NEGATIVE_NUMBER:
                    break

            positives_text = [
                sentence_texts[idx]
                for idx in positives
            ]

            sample = {
                "query": question,
                "pos": positives_text,
                "neg": negatives
            }

            out.write(
                json.dumps(sample, ensure_ascii=False)
            )
            out.write("\n")

            total_samples += 1


print()
print("=" * 60)
print(f"Saved {total_samples} samples")
print(f"Output : {OUTPUT_FILE}")