"""
tests/test_trainer.py
---------------------
Tests for LogisticRegressionTrainer and BaseTrainer interface.
"""
import os
import pytest

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from training.ml_dataset import BenchmarkToMLConverter
from training.split import stratified_split
from training.dataset import MLDataset
from training.trainer import LogisticRegressionTrainer
from training.base_trainer import BaseTrainer

BENCHMARK_PATH = "data/datasets/raw/benchmark_data.json"

EXPECTED_METRIC_KEYS = {
    "precision", "recall", "f1", "auc",
    "ndcg@3", "ndcg@5",
    "recall@1", "recall@3", "recall@5",
}


@pytest.fixture(scope="module")
def splits():
    loader = BenchmarkDatasetLoader(BENCHMARK_PATH)
    dataset = loader.load()
    converter = BenchmarkToMLConverter()
    examples = converter.convert(dataset)
    train_ex, val_ex, test_ex = stratified_split(examples, seed=42)
    return (
        MLDataset(train_ex),
        MLDataset(val_ex),
        MLDataset(test_ex),
    )


@pytest.fixture(scope="module")
def trained_trainer(splits):
    train_ds, val_ds, _ = splits
    trainer = LogisticRegressionTrainer(C=1.0, max_iter=500, seed=42)
    trainer.fit(train_ds, val_ds)
    return trainer


# ---------------------------------------------------------------------------
# BaseTrainer interface
# ---------------------------------------------------------------------------

def test_trainer_is_base_trainer():
    """LogisticRegressionTrainer must implement BaseTrainer."""
    assert issubclass(LogisticRegressionTrainer, BaseTrainer)


# ---------------------------------------------------------------------------
# fit() metric keys and value ranges
# ---------------------------------------------------------------------------

def test_fit_returns_all_metric_keys(splits):
    """fit() must return all required metric keys."""
    train_ds, val_ds, _ = splits
    trainer = LogisticRegressionTrainer(C=1.0, max_iter=500, seed=42)
    metrics = trainer.fit(train_ds, val_ds)

    missing = EXPECTED_METRIC_KEYS - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"


def test_fit_metrics_in_valid_range(splits):
    """All returned metric values must be in [0.0, 1.0]."""
    train_ds, val_ds, _ = splits
    trainer = LogisticRegressionTrainer(C=1.0, max_iter=500, seed=42)
    metrics = trainer.fit(train_ds, val_ds)

    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric '{k}' = {v} is out of [0, 1]"


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------

def test_predict_proba_length(trained_trainer):
    """predict_proba must return one probability per chunk."""
    query  = "What is the governing law?"
    chunks = [
        "This agreement is governed by Delaware law.",
        "The cat sat on the mat.",
        "Party A shall pay fees within 30 days.",
    ]
    probs = trained_trainer.predict_proba(query, chunks)
    assert len(probs) == len(chunks)


def test_predict_proba_range(trained_trainer):
    """All probabilities must be in [0.0, 1.0]."""
    query  = "What is the termination clause?"
    chunks = [
        "Either party may terminate with 30 days notice.",
        "The company sells software products.",
    ]
    probs = trained_trainer.predict_proba(query, chunks)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability {p} out of [0, 1]"


def test_predict_proba_empty(trained_trainer):
    """predict_proba on empty chunk list must return empty list."""
    probs = trained_trainer.predict_proba("some query", [])
    assert probs == []


def test_predict_proba_before_fit_raises():
    """predict_proba before fit() must raise RuntimeError."""
    trainer = LogisticRegressionTrainer()
    with pytest.raises(RuntimeError):
        trainer.predict_proba("query", ["chunk"])


# ---------------------------------------------------------------------------
# save() / load() round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(trained_trainer, tmp_path):
    """Loaded model must produce identical predictions to the original."""
    model_path = str(tmp_path / "lr_router.pkl")
    trained_trainer.save(model_path)
    assert os.path.exists(model_path)

    loaded = LogisticRegressionTrainer.load(model_path)

    query  = "What are the confidentiality obligations?"
    chunks = [
        "Confidential information shall not be disclosed to third parties.",
        "The weather today is sunny.",
        "Employee shall maintain secrecy of trade secrets.",
    ]
    orig_probs   = trained_trainer.predict_proba(query, chunks)
    loaded_probs = loaded.predict_proba(query, chunks)

    assert len(orig_probs) == len(loaded_probs)
    for o, l in zip(orig_probs, loaded_probs):
        assert abs(o - l) < 1e-8, f"Saved/loaded proba mismatch: {o} vs {l}"


def test_save_before_fit_raises(tmp_path):
    """save() on untrained model must raise RuntimeError."""
    trainer = LogisticRegressionTrainer()
    with pytest.raises(RuntimeError):
        trainer.save(str(tmp_path / "bad.pkl"))
