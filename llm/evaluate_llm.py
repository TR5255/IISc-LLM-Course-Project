import json
import os
import time

from rag_pipeline import LegalRAG


# ============================================================
# Paths
# ============================================================

CONTRACT_INDEX = (
    "/kaggle/input/datasets/madhavkumar244/"
    "madhavscuadhn/test_contract_index.json"
)

OUTPUT_FILE = (
    "/kaggle/working/llm_evaluation_results.jsonl"
)

NUM_QUESTIONS = 1244


# ============================================================
# Load dataset
# ============================================================

with open(
    CONTRACT_INDEX,
    "r",
    encoding="utf-8"
) as f:

    contracts = json.load(f)


print("=" * 70)
print("LLM / RAG EVALUATION")
print("=" * 70)

print(
    f"Contracts available : {len(contracts)}"
)

print(
    f"Questions to test   : {NUM_QUESTIONS}"
)

print()


# ============================================================
# Load existing results if present
# ============================================================

completed_keys = set()

if os.path.exists(OUTPUT_FILE):

    print("Existing result file found.")
    print("Checking completed questions...")

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(line)

                key = (
                    item["contract_index"],
                    item["question_index"]
                )

                completed_keys.add(key)

            except Exception:
                continue


    print(
        f"Already completed : "
        f"{len(completed_keys)} questions"
    )

else:

    print("Starting fresh evaluation.")


print()


# ============================================================
# Load RAG once
# ============================================================

print("Loading RAG system...")

rag = LegalRAG()

print()
print("✓ RAG system ready")
print()


# ============================================================
# Evaluation
# ============================================================

count = len(completed_keys)


with open(
    OUTPUT_FILE,
    "a",
    encoding="utf-8"
) as output:


    for contract_idx, contract in enumerate(
        contracts
    ):

        sentences = contract["sentences"]

        passages = [
            sentence["text"].strip()
            for sentence in sentences
        ]


        for qa_idx, qa in enumerate(
            contract["questions"]
        ):


            # ------------------------------------------------
            # Stop after required number of questions
            # ------------------------------------------------

            if count >= NUM_QUESTIONS:
                break


            # ------------------------------------------------
            # Skip already completed questions
            # ------------------------------------------------

            key = (
                contract_idx,
                qa_idx
            )

            if key in completed_keys:
                continue


            question = qa["question"]

            positives = qa.get(
                "positive_sentence_ids",
                []
            )


            print()
            print("=" * 70)

            print(
                f"QUESTION "
                f"{count + 1}/{NUM_QUESTIONS}"
            )

            print("=" * 70)

            print(question)


            # =================================================
            # Run complete RAG
            # =================================================

            try:

                result = rag.answer(
                    question,
                    passages
                )

                record = {

                    "contract_index":
                        contract_idx,

                    "question_index":
                        qa_idx,

                    "question":
                        question,

                    "positive_sentence_ids":
                        positives,

                    "answer":
                        result["answer"],

                    "retrieved":
                        result["retrieved"],

                    "reranked":
                        result["reranked"]

                }


            except Exception as e:

                print()
                print("ERROR:")
                print(str(e))


                record = {

                    "contract_index":
                        contract_idx,

                    "question_index":
                        qa_idx,

                    "question":
                        question,

                    "positive_sentence_ids":
                        positives,

                    "answer":
                        None,

                    "error":
                        str(e)

                }


            # =================================================
            # Save immediately
            # =================================================

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            output.flush()


            completed_keys.add(key)

            count += 1


            # =================================================
            # Print answer
            # =================================================

            print()
            print("-" * 70)
            print("LLM ANSWER")
            print("-" * 70)

            if record["answer"] is not None:

                print(
                    record["answer"]
                )

            else:

                print(
                    "FAILED:",
                    record.get("error")
                )


            # =================================================
            # Print reranked evidence
            # =================================================

            if record.get("reranked"):

                print()
                print("-" * 70)
                print("RERANKED EVIDENCE")
                print("-" * 70)


                for rank, item in enumerate(
                    record["reranked"],
                    1
                ):

                    print()

                    print(
                        f"[{rank}] "
                        f"sentence_index="
                        f"{item['index']} "
                        f"reranker_score="
                        f"{item['reranker_score']:.4f}"
                    )

                    print(
                        item["text"]
                    )


            # =================================================
            # Small delay
            # =================================================

            time.sleep(1)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("LLM EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Questions processed : {count}"
)

print(
    f"Results saved to    : {OUTPUT_FILE}"
)