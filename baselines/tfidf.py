from typing import List
from models.scorer import BaseScorer

class TFIDFScorer(BaseScorer):
    """TF-IDF sentence-level scoring baseline."""
    
    def score(self, question: str, chunks: List[str]) -> List[float]:
        # Simple simulated TF-IDF:
        # Penalizes common words in query if they appear in too many chunks.
        query_words = set(question.lower().split())
        if not query_words:
            return [0.0] * len(chunks)
            
        # 1. Document frequency (DF) of words across chunks
        df = {}
        for word in query_words:
            df[word] = sum(1 for chunk in chunks if word in chunk.lower())
            
        N = len(chunks)
        scores = []
        for chunk in chunks:
            chunk_words = chunk.lower().split()
            chunk_score = 0.0
            
            for word in query_words:
                tf = chunk_words.count(word)
                if tf > 0:
                    idf = max(0.1, N / (df[word] + 1))  # Simple mock IDF
                    chunk_score += tf * idf
                    
            scores.append(chunk_score)
            
        # Normalize scores to [0.0, 1.0] if there is any non-zero score
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
            
        return scores
