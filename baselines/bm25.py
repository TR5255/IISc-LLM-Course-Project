from typing import List
from models.scorer import BaseScorer

class BM25Scorer(BaseScorer):
    """BM25 term-overlap based scoring baseline."""
    
    def score(self, question: str, chunks: List[str]) -> List[float]:
        # Simple simulated BM25 score based on word overlap:
        # Count occurrences of query words in each chunk, normalized.
        query_words = set(question.lower().split())
        if not query_words:
            return [0.0] * len(chunks)
            
        scores = []
        for chunk in chunks:
            chunk_words = chunk.lower().split()
            overlap_count = sum(1 for word in chunk_words if word in query_words)
            # A mock normalization to keep score in [0.0, 1.0]
            scores.append(min(overlap_count / len(query_words), 1.0))
            
        return scores
