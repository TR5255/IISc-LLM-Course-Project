"""
tests/test_external_datasets.py
---------------------------------
Unit tests for external dataset integration (CUAD, LexGLUE, UnifiedRouterDataset).
All tests use in-memory synthetic fixtures — no external network calls or raw files needed.
"""
import json
import pytest
from typing import List

from training.ml_dataset import MLTrainingExample
from data.external.cuad.cuad_loader import CUADLoader
from data.external.lexglue.lexglue_loader import LexGLUELoader
from data.datasets.external_loader import (
    CUADExternalLoader,
    LexGLUEExternalLoader,
    UnifiedRouterDataset,
    ExternalDatasetLoader,
)
from data.datasets.external_statistics import generate_external_report


@pytest.fixture
def synthetic_cuad_dict(tmp_path):
    """Creates a temporary synthetic CUAD-format JSON file."""
    cuad_data = {
        "data": [
            {
                "title": "Test Contract 01",
                "paragraphs": [
                    {
                        "context": "This Agreement is entered into by Alice and Bob. Governing law is Delaware.",
                        "qas": [
                            {
                                "question": "What is the governing law?",
                                "answers": [
                                    {"answer_start": 57, "text": "Delaware"}
                                ],
                            },
                            {
                                "question": "Is there an NDA clause?",
                                "answers": [],  # No answer -> negative example
                            },
                        ],
                    }
                ],
            }
        ]
    }
    file_path = tmp_path / "cuad_fixture.json"
    file_path.write_text(json.dumps(cuad_data), encoding="utf-8")
    return str(file_path)


def test_cuad_loader_with_fixture(synthetic_cuad_dict):
    """Verify CUADLoader correctly converts synthetic CUAD JSON to MLTrainingExample list."""
    loader = CUADLoader()
    examples = loader.load(synthetic_cuad_dict)
    assert len(examples) == 2
    assert all(isinstance(ex, MLTrainingExample) for ex in examples)

    # First QA: positive match
    pos_ex = [ex for ex in examples if ex.label == 1]
    assert len(pos_ex) == 1
    assert pos_ex[0].query == "What is the governing law?"
    assert pos_ex[0].relevance_score == 3
    assert pos_ex[0].difficulty == "external"

    # Second QA: negative match
    neg_ex = [ex for ex in examples if ex.label == 0]
    assert len(neg_ex) == 1
    assert neg_ex[0].query == "Is there an NDA clause?"
    assert neg_ex[0].relevance_score == 0


def test_cuad_label_validity(synthetic_cuad_dict):
    """Verify CUAD labels and relevance scores adhere to expected ranges."""
    loader = CUADLoader()
    examples = loader.load(synthetic_cuad_dict)
    for ex in examples:
        assert ex.label in {0, 1}
        assert ex.relevance_score in {0, 3}
        assert ex.total_chunks >= 1
        assert ex.chunk_tokens > 0


def test_lexglue_compatible_task(tmp_path):
    """Verify LexGLUELoader loads local synthetic EURLEX JSONL fixture."""
    raw_dir = tmp_path / "lexglue" / "raw"
    task_dir = raw_dir / "eurlex"
    task_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = task_dir / "train.jsonl"
    jsonl_file.write_text(
        json.dumps({"text": "European Union trade regulation clause.", "label": [1, 0]}) + "\n" +
        json.dumps({"text": "Local state tax policy document.", "label": [0]}) + "\n",
        encoding="utf-8",
    )

    loader = LexGLUELoader(use_hf=False)
    examples = loader.load("eurlex", data_dir=str(raw_dir))
    assert len(examples) == 2
    assert examples[0].label == 1
    assert examples[0].relevance_score == 1
    assert examples[1].label == 0
    assert examples[1].relevance_score == 0


def test_lexglue_incompatible_task_skips_gracefully():
    """Verify incompatible tasks (e.g. SCOTUS) return [] and log warning without crashing."""
    loader = LexGLUELoader(use_hf=False)
    examples = loader.load("scotus")
    assert examples == []


class DummyExternalLoader(ExternalDatasetLoader):
    """Dummy loader fixture for testing UnifiedRouterDataset."""

    def __init__(self, prefix: str, count: int):
        self.prefix = prefix
        self.count = count

    def load(self) -> List[MLTrainingExample]:
        return [
            MLTrainingExample(
                doc_id=f"{self.prefix}__doc_{i}",
                chunk_id=i,
                query=f"Query {i}",
                chunk_text=f"Sample text chunk {i}",
                label=i % 2,
                relevance_score=3 if (i % 2) else 0,
                difficulty="external",
                chunk_pos=i,
                total_chunks=self.count,
                chunk_tokens=5,
            )
            for i in range(self.count)
        ]

    def source_name(self) -> str:
        return self.prefix


def test_unified_loader_merges_sources():
    """Verify UnifiedRouterDataset aggregates multiple loaders correctly."""
    l1 = DummyExternalLoader("cuad", 3)
    l2 = DummyExternalLoader("lexglue", 2)
    unified = UnifiedRouterDataset([l1, l2])

    examples = unified.load_all()
    assert len(examples) == 5
    summary = unified.summary(examples)
    assert summary["total_examples"] == 5
    assert summary["unique_documents"] == 5
    assert "cuad" in summary["source_counts"]
    assert "lexglue" in summary["source_counts"]


def test_no_schema_violations(synthetic_cuad_dict):
    """Verify MLTrainingExample field types and invariants across external datasets."""
    loader = CUADLoader()
    examples = loader.load(synthetic_cuad_dict)
    for ex in examples:
        assert isinstance(ex.doc_id, str)
        assert isinstance(ex.chunk_id, int)
        assert isinstance(ex.query, str)
        assert isinstance(ex.chunk_text, str)
        assert isinstance(ex.label, int)
        assert isinstance(ex.relevance_score, int)
        assert isinstance(ex.difficulty, str)


def test_external_statistics_report_generation(synthetic_cuad_dict):
    """Verify external statistics report generation produces non-empty markdown string."""
    loader = CUADLoader()
    examples = loader.load(synthetic_cuad_dict)
    report = generate_external_report(examples, source_name="CUAD Test")
    assert "# External Dataset Statistics Report — CUAD Test" in report
    assert "Total ML training rows" in report
