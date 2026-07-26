from abc import ABC, abstractmethod
from typing import List

class BasePolicy(ABC):
    """Abstract base class for routing policies that filter contexts based on relevance scores."""
    
    @abstractmethod
    def select(self, chunks: List[str], scores: List[float]) -> List[str]:
        """Filters the original chunks based on their scores.

        Args:
            chunks: List of document segment/sentence strings.
            scores: Corresponding relevance scores in [0.0, 1.0].

        Returns:
            A list of selected chunk strings.
        """
        pass


class ThresholdPolicy(BasePolicy):
    """Selects chunks that meet or exceed a set score threshold."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def select(self, chunks: List[str], scores: List[float]) -> List[str]:
        return [chunk for chunk, score in zip(chunks, scores) if score >= self.threshold]


class TopKPolicy(BasePolicy):
    """Selects top K highest-scoring chunks, preserving original document order."""
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, chunks: List[str], scores: List[float]) -> List[str]:
        # Zipper and sort based on score in descending order
        sorted_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        top_k_pairs = sorted_pairs[:self.k]
        
        # Keep original document order to preserve semantic flow
        chunk_order = {c: i for i, c in enumerate(chunks)}
        selected_chunks = [pair[0] for pair in top_k_pairs]
        selected_chunks.sort(key=lambda c: chunk_order[c])
        
        return selected_chunks
