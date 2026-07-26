"""
training/split.py
-----------------
Document-level stratified split for MLTrainingExample lists.

Splits at the **document level** — all chunks belonging to a doc go into the
same partition — and stratifies by difficulty (easy/medium/hard) so the
class balance is preserved across train / val / test.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from training.ml_dataset import MLTrainingExample


def stratified_split(
    examples: List[MLTrainingExample],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[List[MLTrainingExample], List[MLTrainingExample], List[MLTrainingExample]]:
    """
    Split *examples* into train / val / test at the document level, preserving
    the difficulty distribution in each partition.

    Args:
        examples: Flat list of MLTrainingExample (multiple chunks per doc).
        train:    Fraction for training   (default 0.70)
        val:      Fraction for validation (default 0.15)
        test:     Fraction for test       (default 0.15)
        seed:     Random seed for reproducibility.

    Returns:
        (train_examples, val_examples, test_examples)
    """
    assert abs(train + val + test - 1.0) < 1e-6, "train + val + test must equal 1.0"

    rng = random.Random(seed)

    # ---- Build a mapping: doc_id → (difficulty, [examples]) ----------------
    doc_map: Dict[str, List[MLTrainingExample]] = defaultdict(list)
    doc_difficulty: Dict[str, str] = {}
    for ex in examples:
        doc_map[ex.doc_id].append(ex)
        doc_difficulty[ex.doc_id] = ex.difficulty

    # ---- Group doc_ids by difficulty ----------------------------------------
    by_difficulty: Dict[str, List[str]] = defaultdict(list)
    for doc_id, diff in doc_difficulty.items():
        by_difficulty[diff].append(doc_id)

    # ---- Allocate docs to splits (stratified) --------------------------------
    train_docs, val_docs, test_docs = [], [], []
    for diff, doc_ids in by_difficulty.items():
        shuffled = doc_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * train))
        n_val   = max(1, round(n * val))   if n > 1 else 0
        # Remaining goes to test (handles rounding)
        n_test  = n - n_train - n_val

        # Edge case: very small groups
        if n_test < 0:
            n_val  = max(0, n_val + n_test)
            n_test = 0

        train_docs.extend(shuffled[:n_train])
        val_docs.extend(shuffled[n_train:n_train + n_val])
        test_docs.extend(shuffled[n_train + n_val:])

    # ---- Collect examples per split ------------------------------------------
    def _gather(doc_ids: List[str]) -> List[MLTrainingExample]:
        out = []
        for doc_id in doc_ids:
            out.extend(doc_map[doc_id])
        return out

    return _gather(train_docs), _gather(val_docs), _gather(test_docs)
