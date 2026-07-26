# LexGLUE — Legal Benchmark Integration

## Overview

[LexGLUE](https://github.com/legalglue/lexglue) (Chalkidis et al., 2021) is a benchmark for legal language understanding in English, comprising 7 diverse tasks.

- **License**: CC BY-SA 4.0 / Open Data
- **Source**: Hugging Face `datasets.load_dataset("lex_glue", task_name)`

## Task Compatibility Matrix

| Task | Type | Router Compatible? | Notes / Mapping Strategy |
|---|---|---|---|
| **EURLEX** | Multi-label classification | ✅ Partial | EU law documents chunked (~512 tokens); label presence as binary relevance |
| **ECtHR (A)** | Binary classification | ✅ Partial | European Court of Human Rights cases; violation presence mapped to relevance |
| **ECtHR (B)** | Multi-label classification | ✅ Partial | Alleged article violations mapped to binary relevance |
| **UNFAIR-ToS** | Multi-label classification | ✅ Partial | Terms of service clause unfairness classification mapped directly |
| **SCOTUS** | Multi-class topic | ❌ Incompatible | Supreme Court topic classification lacks query/chunk retrieval structure |
| **LEDGAR** | Multi-class classification | ❌ Incompatible | Contract provision classification; no query/chunk routing format |
| **CaseHOLD** | Multiple-choice | ❌ Incompatible | Holding identification from judicial options; incompatible format |

> [!NOTE]
> Incompatible tasks generate a warning and return an empty list `[]` when requested via `LexGLUELoader`. They do not raise exceptions.

## Setup & Download

### Option 1: Via HuggingFace Datasets (Recommended)

```bash
pip install datasets
```

```python
from data.external.lexglue.lexglue_loader import LexGLUELoader

loader = LexGLUELoader(use_hf=True)
examples = loader.load("eurlex")
```

### Option 2: Local JSONL Files

Place JSONL data in `data/external/lexglue/raw/{task_name}/{split}.jsonl` (gitignored).

```python
from data.external.lexglue.lexglue_loader import LexGLUELoader

loader = LexGLUELoader(use_hf=False)
examples = loader.load("eurlex", data_dir="data/external/lexglue/raw")
```

## Schema Mapping

| LexGLUE Field | MLTrainingExample Field | Notes |
|---|---|---|
| Task ID / Index | `doc_id` | e.g. `lexglue__eurlex_42` |
| Task Description | `query` | Standard query prompt per task |
| `text` (chunked) | `chunk_text` | Split into ~512-token chunks |
| `label` | `label` | 1 if relevant label present, else 0 |
| Label presence | `relevance_score` | 1 for positive, 0 for negative (no 2/3 grades) |
| — | `difficulty="external"` | Sentinel difficulty marker |

## Differences from Synthetic Benchmark

- **Relevance Granularity**: Binary presence only (0 or 1, no graded 2/3 levels).
- **Task Types**: Formulated as classification tasks mapped into document context retrieval format.
