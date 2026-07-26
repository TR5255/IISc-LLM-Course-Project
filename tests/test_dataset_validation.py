# test_dataset_validation.py
"""Extended validation tests for the expanded legal benchmark dataset.
These tests supplement `test_dataset.py` and ensure data integrity after
Milestone 5 expansion.
"""
import os
from collections import Counter

from data.datasets.benchmark_loader import BenchmarkDatasetLoader

DATA_PATH = os.path.join("data", "datasets", "raw", "benchmark_data.json")


def test_dataset_integrity():
    loader = BenchmarkDatasetLoader(json_path=DATA_PATH)
    dataset = loader.load()
    assert len(dataset) >= 50, "Benchmark should contain at least 50 items after expansion."

    # Track categories based on document_id prefix
    categories = Counter()
    for item in dataset:
        # Category prefix before '_' in document_id
        cat = item.document_id.split("_")[0]
        categories[cat] += 1

        # Verify each chunk has sequential IDs and unique within the item
        chunk_ids = [c.id for c in item.chunks]
        assert chunk_ids == list(range(len(item.chunks))), (
            f"Item {item.document_id} chunk IDs are not sequential 0-indexed."
        )
        # Verify start/end positions slice the document text correctly
        for c in item.chunks:
            if c.start_position is not None and c.end_position is not None:
                slice_text = item.document_text[c.start_position:c.end_position]
                assert slice_text.strip() == c.text.strip(), (
                    f"Chunk {c.id} of {item.document_id} position slice does not match text."
                )
            # Token count should match whitespace split length
            expected_tokens = len(c.text.split())
            assert c.token_count == expected_tokens, (
                f"Chunk {c.id} of {item.document_id} token_count {c.token_count} "
                f"does not match expected {expected_tokens}."
            )
    # Ensure each required legal category appears at least once
    required_cats = {"nda", "employment", "saas", "service", "vendor", "licensing", "privacy", "dpa", "tos"}
    missing = required_cats - set(categories.keys())
    assert not missing, f"Missing required categories in benchmark: {missing}"

    # Simple distribution sanity checks (optional)
    total_chunks = sum(len(item.chunks) for item in dataset)
    assert total_chunks > 0, "Dataset must contain chunks."

    # Verify relevance scores are within 0-3 (already checked in base tests, but repeat)
    for item in dataset:
        for c in item.chunks:
            assert 0 <= c.relevance_score <= 3, (
                f"Chunk {c.id} of {item.document_id} has invalid relevance_score {c.relevance_score}."
            )

    print("Extended dataset integrity tests passed.")
