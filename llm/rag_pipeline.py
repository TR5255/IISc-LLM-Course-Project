import os
import json
import time
from urllib import response
import numpy as np
import torch

from FlagEmbedding import FlagModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from google import genai
from google.genai import types

from prompt import SYSTEM_PROMPT, build_prompt


# ============================================================
# Paths
# ============================================================

CONTRACT_INDEX = (
    "/kaggle/input/datasets/madhavkumar244/"
    "madhavscuadhn/test_contract_index.json"
)

RETRIEVER_PATH = (
    "Madhav2832005/bge-base-legal-retriever"
)

RERANKER_PATH = (
    "Madhav2832005/bge-base-reranker-finetuned"
)

RETRIEVER_TOP_K = 10
RERANKER_TOP_K = 3

MAX_LENGTH = 384
RERANKER_MAX_LENGTH = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# RAG Pipeline
# ============================================================

class LegalRAG:

    def __init__(self):

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        print("Loading retriever...")

        self.retriever = FlagModel(
            RETRIEVER_PATH,
            devices=(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            ),
            use_fp16=torch.cuda.is_available(),
        )

        print("✓ Retriever loaded")

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        print("Loading reranker...")

        self.reranker_tokenizer = (
            AutoTokenizer.from_pretrained(
                RERANKER_PATH
            )
        )

        self.reranker_model = (
            AutoModelForSequenceClassification
            .from_pretrained(RERANKER_PATH)
        )

        self.reranker_model.to(DEVICE)
        self.reranker_model.eval()

        print("✓ Reranker loaded")

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.llm = genai.Client(
            api_key=api_key
        )

        print("✓ Gemini client loaded")

    # ========================================================
    # Retriever
    # ========================================================

    def retrieve(self, question, passages):

        query_embedding = (
            self.retriever.encode_queries(
                [question],
                max_length=MAX_LENGTH,
            )[0]
        )

        passage_embeddings = (
            self.retriever.encode_corpus(
                passages,
                max_length=MAX_LENGTH,
            )
        )

        passage_embeddings = np.asarray(
            passage_embeddings
        )

        scores = (
            passage_embeddings @ query_embedding
        )

        retrieval_order = np.argsort(-scores)

        top_indices = retrieval_order[
            :RETRIEVER_TOP_K
        ]

        candidates = []

        for idx in top_indices:

            candidates.append({
                "text": passages[idx],
                "retriever_score": float(
                    scores[idx]
                ),
                "index": int(idx),
            })

        return candidates

    # ========================================================
    # Reranker
    # ========================================================

    @torch.no_grad()
    def rerank(self, question, candidates):

        passages = [
            candidate["text"]
            for candidate in candidates
        ]

        inputs = self.reranker_tokenizer(
            [question] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        logits = self.reranker_model(
            **inputs
        ).logits

        scores = (
            logits
            .view(-1)
            .float()
            .cpu()
            .numpy()
        )

        rerank_order = np.argsort(-scores)

        ranked = []

        for i in rerank_order:

            item = candidates[i].copy()

            item["reranker_score"] = float(
                scores[i]
            )

            ranked.append(item)

        return ranked[:RERANKER_TOP_K]

    # ========================================================
    # LLM Generation
    # ========================================================

    def generate(self, question, passages):

        prompt = build_prompt(
        question,
        [p["text"] for p in passages],
    )

        while True:

            try:

                response = self.llm.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),
                    max_output_tokens=512,
                ),
            )

                return response.text

            except Exception as e:

                error_text = str(e)

                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                    print(
                    "\nGemini quota reached. "
                    "Waiting 25 seconds..."
                )

                    time.sleep(25)

                else:

                    raise

    # ========================================================
    # Complete RAG pipeline
    # ========================================================

    def answer(self, question, passages):

        # Stage 1: retrieve
        candidates = self.retrieve(
            question,
            passages,
        )

        # Stage 2: rerank
        ranked = self.rerank(
            question,
            candidates,
        )

        # Stage 3: generate
        answer = self.generate(
            question,
            ranked,
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved": candidates,
            "reranked": ranked,
        }


# ============================================================
# Load actual CUAD test contract
# ============================================================

def load_test_contract():

    with open(
        CONTRACT_INDEX,
        "r",
        encoding="utf-8"
    ) as f:

        contracts = json.load(f)

    print(
        f"Loaded {len(contracts)} contracts"
    )

    # First contract for initial test
    contract = contracts[0]

    sentences = contract["sentences"]

    passages = [
        sentence["text"].strip()
        for sentence in sentences
    ]

    questions = contract["questions"]

    return contract, passages, questions


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    contract,passages, questions = load_test_contract()

    qa = questions[0]

    question = qa["question"]

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    rag = LegalRAG()

    result = rag.answer(
        question,
        passages
    )

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print()
    print("=" * 70)
    print("RERANKED PASSAGES")
    print("=" * 70)

    for i, item in enumerate(
        result["reranked"],
        1
    ):

        print()

        print(
            f"[{i}] "
            f"retriever_score="
            f"{item['retriever_score']:.4f} "
            f"reranker_score="
            f"{item['reranker_score']:.4f}"
        )

        print(item["text"])