# Smart AI Router: Dataset Expansion Guide

This developer guide describes how to expand the Legal QA benchmark dataset or integrate new custom datasets (e.g., additional LexGLUE tasks, CUAD contractual subsets, or proprietary legal corpuses) into the Smart AI Router benchmark pipeline.

---

## 1. Dataset JSON Schema Definition

Every dataset loaded into the primary evaluation suite must conform to the strict schema defined in `data/datasets/benchmark_schema.py`. The raw dataset files must be saved under `data/datasets/raw/` in JSON format.

Below is the required JSON structure for each document item:

```json
[
  {
    "document_id": "unique_doc_identifier_01",
    "document_text": "Full text of the legal contract or document...",
    "question": "What standard must be met for governing law?",
    "difficulty": "medium",
    "chunks": [
      {
        "id": 0,
        "text": "First chunk or sentence of the document text.",
        "start_char": 0,
        "end_char": 45,
        "token_count": 9,
        "relevance_score": 0,
        "is_relevant": false
      },
      {
        "id": 1,
        "text": "Governing law of this contract shall be Delaware law.",
        "start_char": 46,
        "end_char": 99,
        "token_count": 10,
        "relevance_score": 3,
        "is_relevant": true
      }
    ],
    "answer": "Delaware law"
  }
]
```

### Graded Relevance Score Schema (0–3)
- **0**: Irrelevant. The segment has no relation to the question.
- **1**: Marginally Relevant. Mention of related concepts, but insufficient to answer.
- **2**: Relevant. Contains secondary or supportive context.
- **3**: Gold Standard / Highly Relevant. This segment directly answers the user query.

`is_relevant` is a binary boolean flag that must be set to `true` if `relevance_score >= 1`.

---

## 2. Subclassing BenchmarkDatasetLoader

To load custom file layouts or transform external legal exports on-the-fly, subclass the dataset loader from `data/datasets/benchmark_loader.py`:

```python
from data.datasets.benchmark_loader import BenchmarkDatasetLoader
from data.datasets.benchmark_schema import BenchmarkQAItem, BenchmarkChunk

class CustomLegalLoader(BenchmarkDatasetLoader):
    def load(self) -> List[BenchmarkQAItem]:
        # Implement custom file parsing or API fetching here
        # Return a list of BenchmarkQAItem objects mapped to correct schemas
        pass
```

Register your custom dataset path in `ui/backend/app.py` or specify the `--source` parameter when executing statistics tools:

```bash
PYTHONPATH=. python -m data.datasets.dataset_statistics --source data/datasets/raw/my_custom_dataset.json
```

---

## 3. Training & Feature Extraction Adaptations

If adding a new dataset to train the Logistic Regression baseline (`training/trainer.py`) or Neural Router (`models/neural_router.py`), ensure you execute the ML serialization pipeline to extract training matrices:

1. **Feature Engineering**: Feature extractor vectors are computed using Trigrams, BM25 similarities, TF-IDF matching, and chunk meta positions (see `training/ml_dataset.py`).
2. **Re-Train baseline models**:
   ```bash
   PYTHONPATH=. python -m training.train --dataset my_custom_dataset.json
   ```

---

## 4. Verification Steps & Schema Safety

To verify dataset structural integrity and avoid schema runtime errors during benchmarks, run the automated test suite:

```bash
PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_validation.py
```

This verifies that:
- Chunk character indices match exact substring bounds within `document_text`.
- Token counts are non-negative and properly initialized.
- Graded relevance scores are bounded between `0` and `3`.
