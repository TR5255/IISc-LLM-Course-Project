"""
tests/test_ml_dataset.py
-------------------------
Tests for the ML data conversion, stratified splitting, and feature extraction.
"""
import pytest
import numpy as np

from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from training.ml_dataset import BenchmarkToMLConverter, save_jsonl, load_jsonl, MLTrainingExample
from training.split import stratified_split
from training.dataset import MLDataset
from training.features import FEATURE_DIM


BENCHMARK_PATH = "data/datasets/raw/benchmark_data.json"


@pytest.fixture(scope="module")
def benchmark_dataset():
    loader = BenchmarkDatasetLoader(BENCHMARK_PATH)
    return loader.load()


@pytest.fixture(scope="module")
def all_examples(benchmark_dataset):
    converter = BenchmarkToMLConverter()
    return converter.convert(benchmark_dataset)


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------

def test_conversion_row_count(benchmark_dataset, all_examples):
    """Total rows == sum of chunk counts across all benchmark items."""
    expected = sum(len(item.chunks) for item in benchmark_dataset.items)
    assert len(all_examples) == expected, (
        f"Expected {expected} rows, got {len(all_examples)}"
    )


def test_conversion_label_validity(all_examples):
    """Every example must have a binary label (0 or 1)."""
    for ex in all_examples:
        assert ex.label in (0, 1), f"Invalid label {ex.label} for {ex.doc_id}/{ex.chunk_id}"


def test_conversion_graded_relevance_preserved(all_examples):
    """relevance_score must always be in {0, 1, 2, 3} and not discarded."""
    for ex in all_examples:
        assert ex.relevance_score in (0, 1, 2, 3), (
            f"Invalid relevance_score {ex.relevance_score} for {ex.doc_id}/{ex.chunk_id}"
        )


def test_conversion_label_graded_alignment(all_examples):
    """Binary label and graded score must be consistent."""
    for ex in all_examples:
        if ex.label == 1:
            assert ex.relevance_score > 0, (
                f"Relevant chunk {ex.doc_id}/{ex.chunk_id} has relevance_score=0"
            )
        else:
            assert ex.relevance_score == 0, (
                f"Irrelevant chunk {ex.doc_id}/{ex.chunk_id} has relevance_score>0"
            )


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip(all_examples, tmp_path):
    """save_jsonl + load_jsonl must produce identical examples."""
    path = str(tmp_path / "examples.jsonl")
    save_jsonl(all_examples, path)
    loaded = load_jsonl(path)

    assert len(loaded) == len(all_examples)
    for orig, back in zip(all_examples, loaded):
        assert orig.doc_id         == back.doc_id
        assert orig.chunk_id       == back.chunk_id
        assert orig.label          == back.label
        assert orig.relevance_score == back.relevance_score


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------

def test_split_sizes(all_examples):
    """Train + val + test must equal total example count."""
    train, val, test = stratified_split(all_examples, train=0.70, val=0.15, test=0.15)
    assert len(train) + len(val) + len(test) == len(all_examples), (
        "Split sizes do not sum to total"
    )


def test_split_no_doc_leakage(all_examples):
    """No document should appear in more than one split."""
    train, val, test = stratified_split(all_examples)
    train_docs = {e.doc_id for e in train}
    val_docs   = {e.doc_id for e in val}
    test_docs  = {e.doc_id for e in test}

    assert train_docs.isdisjoint(val_docs),  "Doc leakage between train and val"
    assert train_docs.isdisjoint(test_docs), "Doc leakage between train and test"
    assert val_docs.isdisjoint(test_docs),   "Doc leakage between val and test"


def test_split_reproducibility(all_examples):
    """Same seed must produce identical splits."""
    t1, v1, te1 = stratified_split(all_examples, seed=42)
    t2, v2, te2 = stratified_split(all_examples, seed=42)
    assert [e.doc_id for e in t1] == [e.doc_id for e in t2]
    assert [e.doc_id for e in v1] == [e.doc_id for e in v2]


# ---------------------------------------------------------------------------
# Feature matrix tests
# ---------------------------------------------------------------------------

def test_feature_matrix_shape(all_examples):
    """Feature matrix must have shape (N, FEATURE_DIM) with FEATURE_DIM=7."""
    train, val, _ = stratified_split(all_examples)
    ds = MLDataset(train)
    X, y_bin, y_grad = ds.get_feature_matrix()

    assert X.shape == (len(train), FEATURE_DIM), (
        f"Expected ({len(train)}, {FEATURE_DIM}), got {X.shape}"
    )
    assert y_bin.shape  == (len(train),), "y_binary shape mismatch"
    assert y_grad.shape == (len(train),), "y_graded shape mismatch"


def test_feature_values_bounded(all_examples):
    """All feature values must be in [0.0, 1.0]."""
    train, _, _ = stratified_split(all_examples)
    ds = MLDataset(train)
    X, _, _ = ds.get_feature_matrix()
    assert np.all(X >= 0.0) and np.all(X <= 1.0), "Feature values out of [0, 1]"


def test_both_labels_returned(all_examples):
    """get_feature_matrix must return both y_binary and y_graded."""
    train, _, _ = stratified_split(all_examples)
    ds = MLDataset(train)
    _, y_bin, y_grad = ds.get_feature_matrix()

    assert set(np.unique(y_bin)).issubset({0, 1}), "y_binary contains values outside {0,1}"
    assert set(np.unique(y_grad)).issubset({0, 1, 2, 3}), "y_graded contains values outside {0,1,2,3}"
