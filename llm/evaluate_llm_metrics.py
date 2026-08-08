import json
import re
import numpy as np
import torch

from FlagEmbedding import FlagModel


# ============================================================
# KAGGLE PATHS
# ============================================================

CONTRACT_INDEX = (
    "/kaggle/input/datasets/madhavkumar244/"
    "madhavscuadhn/test_contract_index.json"
)

LLM_RESULTS = (
    "/kaggle/working/llm_evaluation_results.jsonl"
)


# ============================================================
# MODEL
# ============================================================

RETRIEVER_PATH = (
    "Madhav2832005/bge-base-legal-retriever"
)

MAX_LENGTH = 384

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Text utilities
# ============================================================

def tokenize(text):

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LLM / RAG ANSWER-LEVEL EVALUATION")
    print("=" * 70)

    # ========================================================
    # Check files
    # ========================================================

    print()
    print("Checking files...")

    print(
        f"Contract index : {CONTRACT_INDEX}"
    )

    print(
        f"LLM results    : {LLM_RESULTS}"
    )

    import os

    if not os.path.exists(CONTRACT_INDEX):

        raise FileNotFoundError(
            f"\nContract index not found:\n"
            f"{CONTRACT_INDEX}"
        )

    if not os.path.exists(LLM_RESULTS):

        raise FileNotFoundError(
            f"\nLLM results not found:\n"
            f"{LLM_RESULTS}\n\n"
            f"Make sure the file was downloaded "
            f"from Kaggle to /kaggle/working/."
        )

    # ========================================================
    # Load CUAD test index
    # ========================================================

    print()
    print("Loading CUAD test index...")

    with open(
        CONTRACT_INDEX,
        "r",
        encoding="utf-8"
    ) as f:

        contracts = json.load(f)

    print(
        f"Contracts loaded : {len(contracts)}"
    )

    # ========================================================
    # Build ground-truth lookup
    # ========================================================

    ground_truth = {}

    total_gt_questions = 0
    answerable_gt_questions = 0

    for contract_idx, contract in enumerate(
        contracts
    ):

        sentences = contract["sentences"]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # positive_sentence_ids refer to the POSITION
        # of the sentence in the sentences list.
        #
        # There is no sentence["id"] field.
        # ----------------------------------------------------

        sentence_map = {
            idx: sentence["text"].strip()
            for idx, sentence
            in enumerate(sentences)
        }

        for question_idx, qa in enumerate(
            contract["questions"]
        ):

            total_gt_questions += 1

            positive_ids = [
                int(x)
                for x in qa.get(
                    "positive_sentence_ids",
                    []
                )
            ]

            positive_texts = []

            for sentence_id in positive_ids:

                if sentence_id in sentence_map:

                    positive_texts.append(
                        sentence_map[sentence_id]
                    )

            if positive_ids:

                answerable_gt_questions += 1

            ground_truth[
                (
                    contract_idx,
                    question_idx
                )
            ] = {

                "question":
                    qa["question"],

                "positive_ids":
                    positive_ids,

                "positive_texts":
                    positive_texts

            }

    print(
        f"Ground-truth questions : "
        f"{total_gt_questions}"
    )

    print(
        f"Questions with evidence : "
        f"{answerable_gt_questions}"
    )

    # ========================================================
    # Load LLM results
    # ========================================================

    print()
    print("Loading Gemini evaluation results...")

    results = []

    with open(
        LLM_RESULTS,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            results.append(
                json.loads(line)
            )

    print(
        f"LLM results loaded : {len(results)}"
    )

    # ========================================================
    # Load embedding model
    # ========================================================

    print()
    print("Loading semantic evaluation model...")

    semantic_model = FlagModel(
        RETRIEVER_PATH,
        devices=DEVICE,
        use_fp16=torch.cuda.is_available(),
    )

    print("✓ Semantic model loaded")

    # ========================================================
    # Metric accumulators
    # ========================================================

    total = 0

    successful = 0
    failed = 0

    missing_ground_truth = 0

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    recall_at_1_hits = 0
    recall_at_3_hits = 0

    reciprocal_ranks = []

    answerable_count = 0

    # --------------------------------------------------------
    # Faithfulness / evidence support
    # --------------------------------------------------------

    faithfulness_scores = []

    # --------------------------------------------------------
    # Answer relevance
    # --------------------------------------------------------

    relevance_scores = []

    # --------------------------------------------------------
    # Answer / ground-truth semantic similarity
    # --------------------------------------------------------

    correctness_proxy_scores = []

    # ========================================================
    # Evaluate results
    # ========================================================

    for result_idx, result in enumerate(
        results
    ):

        total += 1

        contract_idx = int(
            result["contract_index"]
        )

        question_idx = int(
            result["question_index"]
        )

        key = (
            contract_idx,
            question_idx
        )

        gt = ground_truth.get(key)

        if gt is None:

            missing_ground_truth += 1

            continue

        answer = result.get(
            "answer"
        )

        # ====================================================
        # Generation success
        # ====================================================

        if not answer:

            failed += 1

            continue

        successful += 1

        # ====================================================
        # Ground truth
        # ====================================================

        positive_ids = gt[
            "positive_ids"
        ]

        positive_texts = gt[
            "positive_texts"
        ]

        positive_set = set(
            positive_ids
        )

        # ====================================================
        # Retrieved / reranked passages
        # ====================================================

        retrieved = result.get(
            "retrieved",
            []
        )

        reranked = result.get(
            "reranked",
            []
        )

        reranked_indices = [

            int(item["index"])

            for item in reranked

        ]

        # ====================================================
        # RETRIEVAL METRICS
        # ====================================================

        if positive_set:

            answerable_count += 1

            # ------------------------------------------------
            # Recall@1
            # ------------------------------------------------

            if len(reranked_indices) >= 1:

                if (
                    reranked_indices[0]
                    in positive_set
                ):

                    recall_at_1_hits += 1

            # ------------------------------------------------
            # Recall@3
            # ------------------------------------------------

            if any(
                idx in positive_set
                for idx in reranked_indices[:3]
            ):

                recall_at_3_hits += 1

            # ------------------------------------------------
            # MRR
            # ------------------------------------------------

            reciprocal_rank = 0.0

            for rank, idx in enumerate(
                reranked_indices,
                start=1
            ):

                if idx in positive_set:

                    reciprocal_rank = (
                        1.0 / rank
                    )

                    break

            reciprocal_ranks.append(
                reciprocal_rank
            )

        # ====================================================
        # SEMANTIC EVALUATION
        # ====================================================

        # ----------------------------------------------------
        # We only evaluate answerable questions for
        # semantic correctness/support because they have
        # actual CUAD evidence.
        # ----------------------------------------------------

        if not positive_texts:

            continue

        # ----------------------------------------------------
        # Encode question and answer
        # ----------------------------------------------------

        embeddings = semantic_model.encode(
            [
                gt["question"],
                answer,
            ],
            max_length=MAX_LENGTH,
        )

        question_embedding = embeddings[0]
        answer_embedding = embeddings[1]

        # ====================================================
        # ANSWER RELEVANCE
        # ====================================================

        relevance = cosine_similarity(
            question_embedding,
            answer_embedding
        )

        relevance_scores.append(
            relevance
        )

        # ====================================================
        # GROUND-TRUTH SEMANTIC SIMILARITY
        # ====================================================

        # Compare answer with each positive
        # contract sentence and take the maximum.
        #
        # This is a semantic-alignment proxy,
        # NOT exact answer accuracy.

        gt_embeddings = semantic_model.encode(
            positive_texts,
            max_length=MAX_LENGTH,
        )

        best_gt_similarity = 0.0

        for gt_embedding in gt_embeddings:

            similarity = cosine_similarity(
                answer_embedding,
                gt_embedding
            )

            best_gt_similarity = max(
                best_gt_similarity,
                similarity
            )

        correctness_proxy_scores.append(
            best_gt_similarity
        )

        # ====================================================
        # FAITHFULNESS / EVIDENCE SUPPORT
        # ====================================================

        # Use the actual Top-3 passages supplied
        # to Gemini.

        top3_texts = [

            item["text"]

            for item in reranked[:3]

        ]

        if top3_texts:

            evidence_embeddings = (
                semantic_model.encode(
                    top3_texts,
                    max_length=MAX_LENGTH,
                )
            )

            best_evidence_similarity = 0.0

            for evidence_embedding in (
                evidence_embeddings
            ):

                similarity = cosine_similarity(
                    answer_embedding,
                    evidence_embedding
                )

                best_evidence_similarity = max(
                    best_evidence_similarity,
                    similarity
                )

            faithfulness_scores.append(
                best_evidence_similarity
            )

        # ====================================================
        # Progress
        # ====================================================

        if (
            (result_idx + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{result_idx + 1}/"
                f"{len(results)}"
            )

    # ========================================================
    # Calculate final metrics
    # ========================================================

    generation_success_rate = (

        successful / total

        if total > 0

        else 0.0

    )

    recall_at_1 = (

        recall_at_1_hits
        / answerable_count

        if answerable_count > 0

        else 0.0

    )

    recall_at_3 = (

        recall_at_3_hits
        / answerable_count

        if answerable_count > 0

        else 0.0

    )

    mrr = mean(
        reciprocal_ranks
    )

    faithfulness = mean(
        faithfulness_scores
    )

    relevance = mean(
        relevance_scores
    )

    correctness_proxy = mean(
        correctness_proxy_scores
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"Questions evaluated      : "
        f"{total}"
    )

    print(
        f"Successful generations   : "
        f"{successful}"
    )

    print(
        f"Failed generations       : "
        f"{failed}"
    )

    print()

    # ========================================================
    # Retrieval
    # ========================================================

    print("-" * 70)
    print("RETRIEVAL / RERANKING")
    print("-" * 70)

    print()

    print(
        f"Recall@1                 : "
        f"{recall_at_1:.4f}"
    )

    print(
        f"Recall@3                 : "
        f"{recall_at_3:.4f}"
    )

    print(
        f"MRR                      : "
        f"{mrr:.4f}"
    )

    # ========================================================
    # LLM
    # ========================================================

    print()
    print("-" * 70)
    print("LLM ANSWER QUALITY")
    print("-" * 70)

    print()

    print(
        f"Faithfulness / "
        f"Evidence Similarity      : "
        f"{faithfulness:.4f}"
    )

    print(
        f"Answer Relevance         : "
        f"{relevance:.4f}"
    )

    print(
        f"Answer / Ground-Truth "
        f"Semantic Similarity      : "
        f"{correctness_proxy:.4f}"
    )

    # ========================================================
    # System
    # ========================================================

    print()
    print("-" * 70)
    print("SYSTEM RELIABILITY")
    print("-" * 70)

    print()

    print(
        f"Generation Success Rate  : "
        f"{generation_success_rate:.4f}"
    )

    # ========================================================
    # Percentage form
    # ========================================================

    print()
    print("=" * 70)
    print("PERCENTAGES")
    print("=" * 70)

    print()

    print(
        f"Recall@1                 : "
        f"{recall_at_1 * 100:.2f}%"
    )

    print(
        f"Recall@3                 : "
        f"{recall_at_3 * 100:.2f}%"
    )

    print(
        f"MRR                      : "
        f"{mrr:.4f}"
    )

    print(
        f"Faithfulness similarity  : "
        f"{faithfulness:.4f}"
    )

    print(
        f"Answer relevance         : "
        f"{relevance:.4f}"
    )

    print(
        f"Ground-truth similarity : "
        f"{correctness_proxy:.4f}"
    )

    print(
        f"Generation success       : "
        f"{generation_success_rate * 100:.2f}%"
    )

    # ========================================================
    # Important interpretation
    # ========================================================

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print()

    print(
        "Semantic metrics are embedding-based proxies."
    )

    print(
        "They are NOT human-annotated answer-accuracy "
        "or factuality judgments."
    )

    print(
        "CUAD provides relevant contract sentences, "
        "not reference natural-language answers."
    )

    print()

    print(
        "Evaluation complete."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()