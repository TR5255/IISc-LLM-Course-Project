"""
training/trainer.py
-------------------
LogisticRegressionTrainer — the Milestone 6 learned router baseline.

Implements BaseTrainer using sklearn LogisticRegression.  Reports both
binary classification metrics (precision, recall, F1, AUC) and ranking
metrics (NDCG@K, Recall@K) computed from graded relevance scores (0-3).
"""
from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training.base_trainer import BaseTrainer
from training.dataset import MLDataset
from training.features import FeatureExtractor


# ---------------------------------------------------------------------------
# Ranking metric helpers (graded relevance, 0-3)
# ---------------------------------------------------------------------------

def _dcg(scores: List[float]) -> float:
    """Discounted Cumulative Gain for an ordered list of relevance scores."""
    return sum(
        score / math.log2(rank + 2)
        for rank, score in enumerate(scores)
    )


def _ndcg_at_k(
    predicted_order: List[int],   # chunk indices sorted by predicted score
    relevance: np.ndarray,        # graded relevance[i] for each chunk i
    k: int,
) -> float:
    """Compute NDCG@K given a predicted ranking and graded relevance labels."""
    topk = predicted_order[:k]
    dcg = _dcg([float(relevance[i]) for i in topk])
    ideal_order = sorted(range(len(relevance)), key=lambda i: relevance[i], reverse=True)
    idcg = _dcg([float(relevance[i]) for i in ideal_order[:k]])
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(
    predicted_order: List[int],
    relevance: np.ndarray,
    k: int,
) -> float:
    """Recall@K: fraction of relevant chunks retrieved in top-K."""
    topk = set(predicted_order[:k])
    relevant = {i for i, r in enumerate(relevance) if r > 0}
    if not relevant:
        return 0.0
    return len(topk & relevant) / len(relevant)


# ---------------------------------------------------------------------------
# Per-document ranking metrics helper
# ---------------------------------------------------------------------------

def _compute_ranking_metrics(
    doc_proba: Dict[str, np.ndarray],       # doc_id → predicted proba array
    doc_relevance: Dict[str, np.ndarray],   # doc_id → graded relevance array
) -> Dict[str, float]:
    """Macro-average NDCG@3, NDCG@5, Recall@1, Recall@3, Recall@5 over docs."""
    ndcg3, ndcg5, r1, r3, r5 = [], [], [], [], []
    for doc_id in doc_proba:
        proba = doc_proba[doc_id]
        rel   = doc_relevance[doc_id]
        order = np.argsort(proba)[::-1].tolist()
        ndcg3.append(_ndcg_at_k(order, rel, 3))
        ndcg5.append(_ndcg_at_k(order, rel, 5))
        r1.append(_recall_at_k(order, rel, 1))
        r3.append(_recall_at_k(order, rel, 3))
        r5.append(_recall_at_k(order, rel, 5))

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "ndcg@3":    _mean(ndcg3),
        "ndcg@5":    _mean(ndcg5),
        "recall@1":  _mean(r1),
        "recall@3":  _mean(r3),
        "recall@5":  _mean(r5),
    }


# ---------------------------------------------------------------------------
# LogisticRegressionTrainer
# ---------------------------------------------------------------------------

class LogisticRegressionTrainer(BaseTrainer):
    """
    Learned router baseline using sklearn LogisticRegression.

    This is the Milestone 6 reference implementation of BaseTrainer.
    Future Qwen/SmolLM trainers will implement the same interface.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        """
        Args:
            C:        Inverse regularisation strength (default 1.0).
            max_iter: Max iterations for solver (default 1000).
            seed:     Random state for reproducibility.
        """
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model: Optional[Any] = None
        self._extractor = FeatureExtractor()

    # -------------------------------------------------------------------------
    # BaseTrainer.fit
    # -------------------------------------------------------------------------
    def fit(self, train_dataset: MLDataset, val_dataset: MLDataset) -> Dict[str, float]:
        """
        Train on *train_dataset*, evaluate on *val_dataset*.

        Returns dict with: precision, recall, f1, auc,
                           ndcg@3, ndcg@5, recall@1, recall@3, recall@5
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_fscore_support,
        )

        # --- Build feature matrices ---
        X_train, y_train_bin, _ = train_dataset.get_feature_matrix()
        X_val,   y_val_bin,   y_val_graded = val_dataset.get_feature_matrix()

        # --- Train ---
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.seed,
            class_weight="balanced",   # handles class imbalance (few relevant chunks)
        )
        self._model.fit(X_train, y_train_bin)

        # --- Binary classification metrics on val split ---
        y_pred     = self._model.predict(X_val)
        y_proba    = self._model.predict_proba(X_val)[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val_bin, y_pred, average="binary", zero_division=0
        )
        auc = float(average_precision_score(y_val_bin, y_proba))

        # --- Ranking metrics (per document, using graded relevance) ---
        # Group val examples by doc_id to compute per-doc NDCG/Recall@K
        doc_proba:     Dict[str, List[float]] = defaultdict(list)
        doc_relevance: Dict[str, List[int]]   = defaultdict(list)
        for i, ex in enumerate(val_dataset.examples):
            doc_proba[ex.doc_id].append(float(y_proba[i]))
            doc_relevance[ex.doc_id].append(int(y_val_graded[i]))

        ranking = _compute_ranking_metrics(
            {k: np.array(v) for k, v in doc_proba.items()},
            {k: np.array(v) for k, v in doc_relevance.items()},
        )

        return {
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "auc":       float(auc),
            **ranking,
        }

    # -------------------------------------------------------------------------
    # BaseTrainer.predict_proba
    # -------------------------------------------------------------------------
    def predict_proba(self, query: str, chunks: List[str]) -> List[float]:
        """Return relevance probability in [0, 1] for each chunk."""
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        if not chunks:
            return []
        # Use a 1-token placeholder per chunk (token count unknown at inference)
        token_counts = [max(1, len(t.split())) for t in chunks]
        X = self._extractor.extract_batch(query, chunks, token_counts)
        return self._model.predict_proba(X)[:, 1].tolist()

    # -------------------------------------------------------------------------
    # BaseTrainer.save / load
    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist model + config to *path* as a pickle file."""
        if self._model is None:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self._model,
            "C":        self.C,
            "max_iter": self.max_iter,
            "seed":     self.seed,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionTrainer":
        """Restore a trained LogisticRegressionTrainer from *path*."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        instance = cls(
            C=payload["C"],
            max_iter=payload["max_iter"],
            seed=payload["seed"],
        )
        instance._model = payload["model"]
        return instance
