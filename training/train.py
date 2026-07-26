"""
training/train.py
-----------------
End-to-end CLI pipeline for the Milestone 6 learned router baseline.

Usage
-----
    python -m training.train [--seed 42] [--output models/saved/lr_router.pkl]

Steps
-----
  1. Load 59-example benchmark
  2. Convert to per-chunk MLTrainingExample rows
  3. Stratified document-level split (70 / 15 / 15)
  4. Extract 7-dim feature matrices (train + val + test)
  5. Train LogisticRegressionTrainer on train split
  6. Evaluate on val split (binary + ranking metrics)
  7. Save JSONL splits to data/training/
  8. Save trained model to --output path
  9. Print full report
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from training.dataset import MLDataset
from training.ml_dataset import BenchmarkToMLConverter, save_jsonl
from training.split import stratified_split
from training.trainer import LogisticRegressionTrainer

BENCHMARK_PATH = "data/datasets/raw/benchmark_data.json"
SPLITS_DIR     = "data/training"
DEFAULT_OUTPUT = "models/saved/lr_router.pkl"


def _print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_metrics(metrics: dict) -> None:
    binary_keys  = ["precision", "recall", "f1", "auc"]
    ranking_keys = ["ndcg@3", "ndcg@5", "recall@1", "recall@3", "recall@5"]

    print("\n  Classification metrics (val split):")
    for k in binary_keys:
        if k in metrics:
            print(f"    {k:<14} {metrics[k]:.4f}")

    print("\n  Ranking metrics (val split, graded relevance 0-3):")
    for k in ranking_keys:
        if k in metrics:
            print(f"    {k:<14} {metrics[k]:.4f}")


def run(seed: int = 42, output: str = DEFAULT_OUTPUT) -> None:
    # ---- 1. Load benchmark --------------------------------------------------
    _print_section("Step 1/8 — Load benchmark")
    loader = BenchmarkDatasetLoader(BENCHMARK_PATH)
    dataset = loader.load()
    print(f"  Loaded {len(dataset)} benchmark items.")

    # ---- 2. Convert to ML rows ----------------------------------------------
    _print_section("Step 2/8 — Convert to per-chunk ML examples")
    converter = BenchmarkToMLConverter()
    examples = converter.convert(dataset)
    print(f"  {len(examples)} MLTrainingExample rows (chunks across all docs).")

    pos = sum(1 for e in examples if e.label == 1)
    neg = len(examples) - pos
    print(f"  Positive (relevant) : {pos}  ({100*pos/len(examples):.1f}%)")
    print(f"  Negative (irrelevant): {neg}  ({100*neg/len(examples):.1f}%)")

    # ---- 3. Stratified split ------------------------------------------------
    _print_section("Step 3/8 — Stratified document-level split (70/15/15)")
    train_ex, val_ex, test_ex = stratified_split(
        examples, train=0.70, val=0.15, test=0.15, seed=seed
    )
    print(f"  Train : {len(train_ex)} rows")
    print(f"  Val   : {len(val_ex)}  rows")
    print(f"  Test  : {len(test_ex)}  rows")

    # ---- 4. Save JSONL splits -----------------------------------------------
    _print_section("Step 4/8 — Save JSONL splits to data/training/")
    os.makedirs(SPLITS_DIR, exist_ok=True)
    save_jsonl(train_ex, os.path.join(SPLITS_DIR, "train.jsonl"))
    save_jsonl(val_ex,   os.path.join(SPLITS_DIR, "val.jsonl"))
    save_jsonl(test_ex,  os.path.join(SPLITS_DIR, "test.jsonl"))
    print(f"  Written: {SPLITS_DIR}/{{train,val,test}}.jsonl")

    # ---- 5. Build datasets --------------------------------------------------
    _print_section("Step 5/8 — Extract 7-dim feature matrices")
    t0 = time.time()
    train_ds = MLDataset(train_ex)
    val_ds   = MLDataset(val_ex)
    test_ds  = MLDataset(test_ex)
    # Trigger feature extraction once to report shape
    X_train, y_b, y_g = train_ds.get_feature_matrix()
    print(f"  Train matrix : {X_train.shape}  (positive={y_b.sum()}, negative={(y_b==0).sum()})")
    elapsed = time.time() - t0
    print(f"  Feature extraction time: {elapsed:.2f}s")

    # ---- 6. Train -----------------------------------------------------------
    _print_section("Step 6/8 — Train LogisticRegressionTrainer")
    trainer = LogisticRegressionTrainer(C=1.0, max_iter=1000, seed=seed)
    t0 = time.time()
    metrics = trainer.fit(train_ds, val_ds)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.2f}s")
    _print_metrics(metrics)

    # ---- 7. Test-set evaluation (no tuning) ---------------------------------
    _print_section("Step 7/8 — Final evaluation on held-out test split")
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    from collections import defaultdict
    from training.trainer import _compute_ranking_metrics

    X_test, y_test_bin, y_test_graded = test_ds.get_feature_matrix()
    y_pred  = trainer._model.predict(X_test)
    y_proba = trainer._model.predict_proba(X_test)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_bin, y_pred, average="binary", zero_division=0
    )
    auc = float(average_precision_score(y_test_bin, y_proba))

    doc_proba, doc_rel = defaultdict(list), defaultdict(list)
    for i, ex in enumerate(test_ds.examples):
        doc_proba[ex.doc_id].append(float(y_proba[i]))
        doc_rel[ex.doc_id].append(int(y_test_graded[i]))

    ranking = _compute_ranking_metrics(
        {k: np.array(v) for k, v in doc_proba.items()},
        {k: np.array(v) for k, v in doc_rel.items()},
    )

    test_metrics = {"precision": float(prec), "recall": float(rec),
                    "f1": float(f1), "auc": float(auc), **ranking}

    print("  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:<14} {v:.4f}")

    # ---- 8. Save model ------------------------------------------------------
    _print_section("Step 8/8 — Save model")
    trainer.save(output)
    print(f"  Model saved to: {output}")

    _print_section("Done")
    print(f"  Baseline pipeline complete. Learned router model written to {output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 6: Train the learned router baseline (LogisticRegression)."
    )
    parser.add_argument("--seed",   type=int, default=42,            help="Random seed")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Model output path")
    args = parser.parse_args()

    run(seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
