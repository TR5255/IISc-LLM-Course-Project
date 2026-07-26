# CUAD — Contract Understanding Atticus Dataset

## Overview

[CUAD](https://huggingface.co/datasets/cuad) (Atticus Project, 2021) contains **510 commercial legal contracts** manually annotated with **41 clause-type labels** in a SQuAD-style span format.

- **License**: CC BY 4.0
- **Size**: ~510 contracts, 13,000+ clause annotations
- **Format**: SQuAD 2.0 JSON (`{ "data": [ { "title", "paragraphs": [ { "context", "qas" } ] } ] }`)

## Why CUAD for the Smart AI Router?

CUAD maps naturally to the router training format:
- The **context** paragraph is the document chunk
- The **question** is the clause-type query (e.g. "Is there a governing law clause?")
- The **answer span** identifies the relevant chunk → `label=1, relevance_score=3`
- Chunks with no overlap → `label=0, relevance_score=0`

## Setup

```bash
# From the project root:
bash data/external/cuad/download_cuad.sh
```

This downloads `CUAD_v1.json` to `data/external/cuad/raw/` (gitignored).

## Schema Mapping

| CUAD field | MLTrainingExample field | Notes |
|---|---|---|
| `title` | `doc_id` | |
| `context` | `chunk_text` | Context split into ~512-token chunks |
| `question` | `query` | 41 clause-type questions per contract |
| answer span present | `label=1, relevance_score=3` | Chunk overlaps answer span |
| no answer span | `label=0, relevance_score=0` | |
| — | `difficulty="external"` | CUAD has no difficulty field |

## Differences from Synthetic Benchmark

| Property | Synthetic Benchmark | CUAD |
|---|---|---|
| Graded relevance | 0, 1, 2, 3 | Only 0 or 3 (absent/present) |
| Difficulty levels | easy / medium / hard | Not available (use "external") |
| Questions per doc | 1 | Up to 41 |
| Answer field | Free text | Span coordinate only |
| Source type | Synthetic | Real commercial contracts |

## Known Limitations

- CUAD clause questions are closed-form ("Is there a…?") — different from open QA
- Some contracts have very long contexts; chunking strategy matters
- No graded relevance between 0 and 3 — future work can infer partial match scores

## Usage

```python
from data.external.cuad.cuad_loader import CUADLoader

loader = CUADLoader()
examples = loader.load("data/external/cuad/raw/CUAD_v1.json")
print(f"{len(examples)} MLTrainingExample rows loaded from CUAD")
```
