"""
training/dataset.py
-------------------
MLDataset: wraps a list of MLTrainingExample and provides feature matrices
for training/evaluation use.

Both binary labels (for classification loss) and graded relevance scores
(for ranking metrics) are exposed.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from training.features import FeatureExtractor, FEATURE_DIM
from training.ml_dataset import MLTrainingExample


class MLDataset:
    """
    Dataset wrapper for ML training.

    Provides:
      - Sequence interface (__len__ / __getitem__) over MLTrainingExample
      - get_feature_matrix() → (X, y_binary, y_graded) numpy arrays
    """

    def __init__(self, examples: List[MLTrainingExample]):
        self.examples = examples
        self._extractor = FeatureExtractor()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MLTrainingExample:
        return self.examples[idx]

    # -------------------------------------------------------------------------
    # Feature matrix construction
    # -------------------------------------------------------------------------
    def get_feature_matrix(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix by grouping examples into documents and calling
        FeatureExtractor.extract_batch() for accurate IDF-based scores.

        Returns
        -------
        X         : np.ndarray, shape (N, FEATURE_DIM)  — float32 features
        y_binary  : np.ndarray, shape (N,)              — int  {0, 1}
        y_graded  : np.ndarray, shape (N,)              — int  {0, 1, 2, 3}
        """
        from collections import defaultdict

        # Group examples by (doc_id, query) so batch extraction is accurate
        doc_groups: dict = defaultdict(list)
        for i, ex in enumerate(self.examples):
            key = (ex.doc_id, ex.query)
            doc_groups[key].append((i, ex))

        X = np.zeros((len(self.examples), FEATURE_DIM), dtype=np.float32)
        y_binary = np.zeros(len(self.examples), dtype=np.int32)
        y_graded = np.zeros(len(self.examples), dtype=np.int32)

        for (doc_id, query), indexed_examples in doc_groups.items():
            chunks      = [ex.chunk_text  for _, ex in indexed_examples]
            token_counts = [ex.chunk_tokens for _, ex in indexed_examples]

            batch_features = self._extractor.extract_batch(query, chunks, token_counts)

            for local_pos, (global_idx, ex) in enumerate(indexed_examples):
                X[global_idx]        = batch_features[local_pos]
                y_binary[global_idx] = ex.label
                y_graded[global_idx] = ex.relevance_score

        return X, y_binary, y_graded
