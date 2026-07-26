"""
models/neural_router.py
------------------------
NeuralRouter implementation wrapping small neural transformer models or sequence classifiers
to score (query, chunk_text) relevance pairs as a BaseScorer subclass.

Usage
-----
    from models.neural_router import NeuralRouter
    from training.models.transformer_router import TransformerRouterModel

    model_wrapper = TransformerRouterModel(...)
    router = NeuralRouter(model_wrapper=model_wrapper)
    scores = router.score(question="What is the governing law?", chunks=["Chunk 1...", "Chunk 2..."])
"""
from __future__ import annotations

from typing import Any, List, Optional
from models.scorer import BaseScorer
from models.model_adapter import ModelAdapter


class NeuralRouter(BaseScorer):
    """
    BaseScorer adapter wrapping neural sequence classification / router models.

    Parameters
    ----------
    model_wrapper : Any
        An instance of TransformerRouterModel or callable model that accepts
        pairs of (query, chunk_text) and returns raw output scores/logits.
    adapter : ModelAdapter, optional
        Adapter to map raw model outputs to normalized [0.0, 1.0] scores.
    batch_size : int
        Batch size for inference scoring.
    """

    def __init__(
        self,
        model_wrapper: Optional[Any] = None,
        adapter: Optional[ModelAdapter] = None,
        batch_size: int = 16,
    ):
        self.model_wrapper = model_wrapper
        self.adapter = adapter or ModelAdapter(output_type="binary_logit")
        self.batch_size = batch_size

    def score(self, question: str, chunks: List[str]) -> List[float]:
        """
        Calculate importance scores for a list of document chunks given a query/question.

        Returns:
            List of float scores bounded in [0.0, 1.0].
        """
        if not chunks:
            return []

        if self.model_wrapper is None:
            # Fallback: if no underlying neural model wrapper is provided, return baseline 0.5
            return [0.5 for _ in chunks]

        scores: List[float] = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_pairs = [(question, chunk) for chunk in batch_chunks]

            # Query the underlying neural model wrapper
            if hasattr(self.model_wrapper, "predict_batch"):
                raw_outputs = self.model_wrapper.predict_batch(batch_pairs)
            elif callable(self.model_wrapper):
                raw_outputs = self.model_wrapper(batch_pairs)
            else:
                raw_outputs = [0.5 for _ in batch_chunks]

            raw_outputs = raw_outputs[: len(batch_chunks)]
            batch_scores = self.adapter.adapt_batch(raw_outputs)
            scores.extend(batch_scores)

        return scores
