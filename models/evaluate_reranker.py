from FlagEmbedding import FlagModel
import json
import os
import contextlib
import numpy as np
from tqdm import tqdm
import torch

RETRIEVER_PATH = "Madhav2832005/bge-base-legal-retriever"
RERANKER_PATH = "Madhav2832005/bge-base-reranker-finetuned"
TOP_K = 10
MAX_LENGTH = 384
CONTRACT_INDEX = "/kaggle/input/datasets/madhavkumar244/madhavscuadhn/test_contract_index.json"
BATCH_SIZE = 256
QUERY_BATCH_SIZE = 64



# ==========================================================
# Suppress FlagEmbedding progress bars
# ==========================================================

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield

from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "Madhav2832005/bge-base-reranker-finetuned",
    use_fp16=True
)

def rerank(question, passages):
    pairs = [[question, p] for p in passages]
    return np.array(reranker.compute_score(pairs))

# ==========================================================
# Main
# ==========================================================

def main():

    print("Loading Retriever...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retriever = FlagModel(
    RETRIEVER_PATH,
    devices="cuda:0",
    use_fp16=True
)

    print("✓ Retriever loaded")

    print()




    

    
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

            sentence_embeddings = retriever.encode_corpus(
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

            query_embeddings = retriever.encode_queries(
                valid_questions,
                batch_size=QUERY_BATCH_SIZE,
                max_length=MAX_LENGTH
            )

        query_embeddings = np.asarray(query_embeddings)

        # ---------------------------------------------
        # Evaluate
        # ---------------------------------------------

        for question_idx, (query_embedding, positives) in enumerate(
    zip(query_embeddings, positive_sets)
):

            total_questions += 1
            
            retrieval_scores = sentence_embeddings @ query_embedding

            retrieval_order = np.argsort(-retrieval_scores)

            top_indices = retrieval_order[:TOP_K]
   
            passages = [
            sentence_texts[idx]
            for idx in top_indices
            ]

            rerank_scores = rerank(valid_questions[question_idx],passages)
            

            
            rerank_scores = np.asarray(rerank_scores)
            rerank_order = np.argsort(-rerank_scores)
      
            ranking = top_indices[rerank_order]
            

           

            
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
    print("Retriever + Reranker Evaluation Results")
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
