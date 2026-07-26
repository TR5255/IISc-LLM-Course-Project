from abc import ABC, abstractmethod
from typing import List

class BaseScorer(ABC):
    """Abstract base class for all scoring engines."""
    
    @abstractmethod
    def score(self, question: str, chunks: List[str]) -> List[float]:
        """Calculates relevance scores for a list of document chunks given a query/question.

        Args:
            question: The incoming user prompt or question.
            chunks: A list of text strings representing document sentences/paragraphs.

        Returns:
            A list of float scores between 0.0 and 1.0.
        """
        pass


class RandomScorer(BaseScorer):
    """Naive scorer returning random values. Helps verify metrics and downstream pipelines flow."""
    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    def score(self, question: str, chunks: List[str]) -> List[float]:
        return [self.rng.random() for _ in chunks]
