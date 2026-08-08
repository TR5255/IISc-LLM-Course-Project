import json
import os
import contextlib
import numpy as np
from tqdm import tqdm
import torch
from FlagEmbedding import FlagModel


# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "Madhav2832005/bge-base-legal-retriever"

CONTRACT_INDEX = "/kaggle/input/datasets/madhavkumar244/madhavscuadhn/test_contract_index.json"

BATCH_SIZE = 256
QUERY_BATCH_SIZE = 64
MAX_LENGTH = 384


# ==========================================================
# Suppress FlagEmbedding progress bars
# ==========================================================

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield


# ==========================================================
# Main
# ==========================================================

def main():

    print("Loading model...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FlagModel(
    MODEL_PATH,
    devices=[device],
    use_fp16=(device != "cpu")
)

    print("✓ Model loaded\n")

    with open(CONTRACT_INDEX, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    print(f"Contracts : {len(contracts)}")
    print()

    total_questions = 0

    recall1 = 0
    recall3 = 0
    recall5 = 0
    recall10 = 0

    mrr = 0.0

    # ======================================================
    # Evaluation
    # ======================================================

    for contract in tqdm(contracts, desc="Evaluating Contracts"):

        sentences = contract["sentences"]
        questions = contract["questions"]

        sentence_texts = [
            s["text"]
            for s in sentences
        ]

        # ---------------------------------------------
        # Encode sentences
        # ---------------------------------------------

        with suppress_output():

            sentence_embeddings = model.encode_corpus(
                sentence_texts,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH
            )

        sentence_embeddings = np.asarray(sentence_embeddings)

        # ---------------------------------------------
        # Collect valid questions
        # ---------------------------------------------

        valid_questions = []
        positive_sets = []

        for qa in questions:

            positives = set(qa["positive_sentence_ids"])

            if len(positives) == 0:
                continue

            valid_questions.append(
                qa["question"]
            )

            positive_sets.append(
                positives
            )

        if len(valid_questions) == 0:
            continue

        # ---------------------------------------------
        # Encode ALL questions together
        # ---------------------------------------------

        with suppress_output():

            query_embeddings = model.encode_queries(
                valid_questions,
                batch_size=QUERY_BATCH_SIZE,
                max_length=MAX_LENGTH
            )

        query_embeddings = np.asarray(query_embeddings)

        # ---------------------------------------------
        # Evaluate
        # ---------------------------------------------

        for query_embedding, positives in zip(
            query_embeddings,
            positive_sets
        ):

            total_questions += 1

            scores = sentence_embeddings @ query_embedding

            ranking = np.argsort(-scores)

            if any(i in positives for i in ranking[:1]):
                recall1 += 1

            if any(i in positives for i in ranking[:3]):
                recall3 += 1

            if any(i in positives for i in ranking[:5]):
                recall5 += 1

            if any(i in positives for i in ranking[:10]):
                recall10 += 1

            for rank, idx in enumerate(ranking):

                if idx in positives:

                    mrr += 1.0 / (rank + 1)

                    break

    # ======================================================
    # Results
    # ======================================================

    print()
    print("=" * 60)
    print("FlagEmbedding Evaluation Results")
    print("=" * 60)
    print()

    print(f"Questions : {total_questions}")
    print()

    print(f"Recall@1  : {recall1 / total_questions:.4f}")
    print(f"Recall@3  : {recall3 / total_questions:.4f}")
    print(f"Recall@5  : {recall5 / total_questions:.4f}")
    print(f"Recall@10 : {recall10 / total_questions:.4f}")
    print(f"MRR        : {mrr / total_questions:.4f}")


if __name__ == "__main__":
    main()