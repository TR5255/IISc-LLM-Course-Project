from typing import List
import numpy as np
from models.scorer import BaseScorer

class CharFrequencyScorer(BaseScorer):
    """Original character frequency-based scoring baseline."""
    
    def score(self, question: str, chunks: List[str]) -> List[float]:
        def get_char_vector(text: str) -> np.ndarray:
            vocab = "abcdefghijklmnopqrstuvwxyz0123456789"
            vec = np.zeros(len(vocab))
            text_lower = text.lower()
            for char in text_lower:
                if char in vocab:
                    idx = vocab.index(char)
                    vec[idx] += 1
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

        if not chunks:
            return []
            
        q_vec = get_char_vector(question)
        scores = []
        for chunk in chunks:
            c_vec = get_char_vector(chunk)
            sim = float(np.dot(q_vec, c_vec))
            scores.append(sim)
            
        return scores


class EmbeddingScorer(BaseScorer):
    """Cosine similarity embedding-based scoring baseline using character 3-grams."""
    
    def score(self, question: str, chunks: List[str]) -> List[float]:
        import math
        from collections import Counter
        
        def tokenize_char_trigrams(text: str) -> List[str]:
            text_clean = text.lower()
            text_clean = " ".join(text_clean.split())
            if len(text_clean) < 3:
                return [text_clean]
            return [text_clean[i:i+3] for i in range(len(text_clean) - 2)]

        def counter_cosine_similarity(c1: Counter, c2: Counter) -> float:
            intersection = set(c1.keys()) & set(c2.keys())
            numerator = sum(c1[x] * c2[x] for x in intersection)
            sum1 = sum(val**2 for val in c1.values())
            sum2 = sum(val**2 for val in c2.values())
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
            if not denominator:
                return 0.0
            return numerator / denominator

        if not chunks:
            return []

        q_trigrams = Counter(tokenize_char_trigrams(question))
        scores = []
        for chunk in chunks:
            c_trigrams = Counter(tokenize_char_trigrams(chunk))
            sim = counter_cosine_similarity(q_trigrams, c_trigrams)
            scores.append(sim)

        return scores

