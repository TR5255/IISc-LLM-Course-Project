"""
models/model_adapter.py
------------------------
Model adapter interface to normalize raw neural network outputs (logits, probabilities,
or regression scalars) into standardized importance scores in [0.0, 1.0] or [0, 3].
"""
from __future__ import annotations

import math
from typing import Any, List, Union


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class ModelAdapter:
    """
    Adapts neural sequence classification or regression outputs into router relevance scores.

    Parameters
    ----------
    output_type : str
        'binary_logit' (2 classes, softmax/sigmoid on positive class),
        'single_logit' (1 linear scalar, sigmoid activation), or
        'probability' (already in [0, 1]).
    """

    def __init__(self, output_type: str = "binary_logit"):
        if output_type not in {"binary_logit", "single_logit", "probability"}:
            raise ValueError(f"Unsupported output_type: '{output_type}'")
        self.output_type = output_type

    def adapt_single(self, raw_output: Any) -> float:
        """
        Convert a single raw prediction output into a normalized float score in [0.0, 1.0].
        """
        if self.output_type == "probability":
            score = float(raw_output)
        elif self.output_type == "single_logit":
            score = _sigmoid(float(raw_output))
        elif self.output_type == "binary_logit":
            # If 2 logits provided [neg_logit, pos_logit]
            if isinstance(raw_output, (list, tuple)) and len(raw_output) >= 2:
                neg_l, pos_l = raw_output[0], raw_output[1]
                diff = pos_l - neg_l
                score = _sigmoid(float(diff))
            else:
                score = _sigmoid(float(raw_output))
        else:
            score = 0.0

        return max(0.0, min(1.0, score))

    def adapt_batch(self, raw_outputs: List[Any]) -> List[float]:
        """
        Convert a batch of raw outputs into normalized float scores.
        """
        return [self.adapt_single(out) for out in raw_outputs]

    def to_graded_relevance(self, normalized_score: float) -> int:
        """
        Map a normalized score in [0.0, 1.0] to a 0–3 graded relevance integer:
          0: < 0.25 (Irrelevant)
          1: 0.25 – 0.50 (Supporting context)
          2: 0.50 – 0.75 (Useful)
          3: >= 0.75 (Essential)
        """
        if normalized_score < 0.25:
            return 0
        elif normalized_score < 0.50:
            return 1
        elif normalized_score < 0.75:
            return 2
        else:
            return 3
