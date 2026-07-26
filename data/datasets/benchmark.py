from typing import Any, Dict, List

class LegalQAItem:
    """Represents a single query-driven context pruning item."""
    def __init__(self, doc_id: str, document_text: str, question: str, gold_sentence_indices: List[int], reference_answer: str):
        self.doc_id = doc_id
        self.document_text = document_text
        self.question = question
        self.gold_sentence_indices = gold_sentence_indices  # Indices of sentences required to answer
        self.reference_answer = reference_answer


class LegalRouterBenchmark:
    """Assembles relevance benchmark dataset instances for evaluation."""
    def __init__(self, items: List[LegalQAItem]):
        self.items = items

    @classmethod
    def get_mock_benchmark(cls) -> "LegalRouterBenchmark":
        """Generates a mock benchmark dataset with legal items for initial research."""
        doc1 = (
            "This NDA is between CorpA and CorpB. "  # index 0
            "Confidential Info must be kept secret. "  # index 1
            "The agreement term is five years. "    # index 2
            "Governing law is New York. "            # index 3
            "Breach of contract will result in injunctions."  # index 4
        )
        
        doc2 = (
            "This Terms of Service applies to AppX. "    # index 0
            "User data is collected for ads. "            # index 1
            "We do not sell data to third parties. "      # index 2
            "Arbitration is required for disputes. "       # index 3
            "Users must be 13 years or older."            # index 4
        )
        
        items = [
            LegalQAItem(
                doc_id="nda_01",
                document_text=doc1,
                question="What is the duration of this agreement?",
                gold_sentence_indices=[2],  # Index of "The agreement term is five years."
                reference_answer="5 years"
            ),
            LegalQAItem(
                doc_id="nda_02",
                document_text=doc1,
                question="Which state governs this agreement?",
                gold_sentence_indices=[3],  # Index of "Governing law is New York."
                reference_answer="New York"
            ),
            LegalQAItem(
                doc_id="tos_01",
                document_text=doc2,
                question="Can a 10 year old legally use AppX?",
                gold_sentence_indices=[4],  # Index of "Users must be 13 years or older."
                reference_answer="No, users must be 13 years or older."
            )
        ]
        return cls(items)
