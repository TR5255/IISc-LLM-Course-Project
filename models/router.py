from typing import List
from models.scorer import BaseScorer
from models.policies import BasePolicy

class SmartAIRouter:
    """The main router routing relevant sentences or paragraphs of long texts to an LLM."""
    def __init__(self, scorer: BaseScorer, policy: BasePolicy):
        self.scorer = scorer
        self.policy = policy

    def route(self, question: str, chunks: List[str]) -> List[str]:
        """Runs the scoring logic and filters the text chunks using the policy."""
        scores = self.scorer.score(question, chunks)
        selected = self.policy.select(chunks, scores)
        return selected
