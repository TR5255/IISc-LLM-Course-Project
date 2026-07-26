"""
training/base_trainer.py
------------------------
Abstract BaseTrainer protocol.

This defines the interface that ALL router trainers must implement —
LogisticRegressionTrainer today, Qwen/SmolLM-based trainers tomorrow.
Keeping the contract here means the CLI and any downstream consumers
never need to know which concrete model they are using.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTrainer(ABC):
    """
    Model-agnostic training interface for neural / learned router models.

    Subclasses
    ----------
    - LogisticRegressionTrainer  (Milestone 6, learned baseline)
    - QwenRouterTrainer          (future Milestone 7+)
    - SmolLMRouterTrainer        (future Milestone 7+)
    """

    @abstractmethod
    def fit(self, train_dataset: Any, val_dataset: Any) -> Dict[str, float]:
        """
        Train the model on *train_dataset* and evaluate on *val_dataset*.

        Returns
        -------
        dict with at minimum:
            precision, recall, f1, auc,
            ndcg@3, ndcg@5,
            recall@1, recall@3, recall@5
        """
        ...

    @abstractmethod
    def predict_proba(self, query: str, chunks: List[str]) -> List[float]:
        """
        Return a relevance probability in [0.0, 1.0] for each chunk.

        Args:
            query:  The question / routing query.
            chunks: List of chunk texts for a single document.

        Returns:
            List[float] of the same length as *chunks*.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTrainer":
        """Restore a trained model from *path*."""
        ...
