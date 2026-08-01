# Smart AI Router

A lightweight, research-oriented Python framework for experimenting with **context routing** — selectively filtering document chunks before sending them to a large language model (LLM). The goal is to reduce token cost and latency without degrading answer quality.

---

## 🚀 Quick Start

Get up and running in **under 1 minute**:

```bash
# 1. Setup environment and install dependencies
./setup.sh

# 2. Launch FastAPI backend and Web UI
./run.sh
```

Or run individual helper scripts:
- `./run_backend.sh` — Launches FastAPI backend server on `http://localhost:8000`.
- `./run_frontend.sh` — Opens Web Dashboard (`ui/frontend/index.html`) in browser.

---

## ⚡ Primary Workflow — Benchmark All

1. Open the Web Dashboard.
2. Click **`[Execute Full Benchmark Suite]`**.
3. The platform automatically benchmarks all implemented routing principles (**BM25**, **TF-IDF**, **Embedding**, etc.) on the standardized **LexGLUE** legal dataset with **Gemini Flash** downstream evaluation.
4. Generates paper-ready academic reports (**Markdown**, **PDF**, **CSV**, **JSON**, and high-res **vector plots**) saved to `data/reports/`.

---

## 🎯 Research Question

> *Can a lightweight local model reduce LLM context by 50–80% while preserving answer quality?*

**Hypothesis**: A small router model (\~0.5B–1B parameters) can learn to rank and filter document chunks from query-chunk relevance signals, producing a compressed context window that is cheaper and faster to send to a large LLM downstream.

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌────────────────┐
│  Router Model  │  ← lightweight (classical baseline or learned neural router)
└────────────────┘
    │  selected chunks only
    ▼
┌─────────────────────┐
│  Large LLM (GPT-4o, │
│  Claude, Qwen…)     │
└─────────────────────┘
    │
    ▼
Final Answer
```

The router operates entirely locally. The LLM sees only the chunks the router deems relevant.

---

## 📂 Project Structure

```
smart-ai-router/
├── data/
│   ├── datasets/
│   │   ├── raw/benchmark_data.json       # 59-item annotated legal QA benchmark
│   │   ├── benchmark_schema.py           # BenchmarkQAItem / Chunk dataclasses
│   │   ├── benchmark_loader.py           # Loads benchmark JSON
│   │   ├── dataset_statistics.py         # Benchmark statistics + markdown report
│   │   └── external_loader.py            # Unified loader for external datasets
│   ├── external/
│   │   ├── cuad/                         # CUAD contract dataset integration
│   │   │   ├── cuad_loader.py            # CUAD JSON → MLTrainingExample
│   │   │   ├── download_cuad.sh          # Download script (data not committed)
│   │   │   └── README.md
│   │   └── lexglue/                      # LexGLUE legal NLP benchmark integration
│   │       ├── lexglue_loader.py         # LexGLUE tasks → MLTrainingExample
│   │       └── README.md
│   ├── loaders/                          # Generic document ingestion utilities
│   └── preprocess/                       # Sentence / character-level chunking
├── baselines/
│   ├── bm25.py                           # BM25 term-frequency scorer
│   ├── tfidf.py                          # TF-IDF cosine similarity scorer
│   └── embedding.py                      # Character trigram embedding scorer
├── models/
│   ├── scorer.py                         # BaseScorer + RandomScorer
│   ├── policies.py                       # ThresholdPolicy, TopKPolicy
│   └── router.py                         # SmartAIRouter (scorer + policy)
├── training/
│   ├── ml_dataset.py                     # MLTrainingExample + BenchmarkToMLConverter
│   ├── split.py                          # Stratified document-level train/val/test split
│   ├── features.py                       # 7-dim feature extractor
│   ├── base_trainer.py                   # Abstract BaseTrainer interface
│   ├── dataset.py                        # MLDataset (feature matrices + graded labels)
│   ├── trainer.py                        # LogisticRegressionTrainer (learned baseline)
│   └── train.py                          # End-to-end CLI training pipeline
├── evaluation/
│   ├── compression.py                    # Token compression ratio
│   ├── retention.py                      # Precision, Recall, F1, NDCG
│   └── llm_eval.py                       # Mock downstream LLM evaluation
├── experiments/
│   ├── configs/                          # YAML experiment configs
│   ├── runs/                             # Run output logs (gitignored)
│   ├── run.py                            # Baseline experiment runner
│   └── analyze.py                        # Sweep result analyser
├── tests/
│   ├── test_dataset.py                   # Benchmark schema and scorer tests
│   ├── test_router.py                    # Router, policy, and metric tests
│   ├── test_ml_dataset.py                # ML conversion, split, and feature tests
│   ├── test_trainer.py                   # LogisticRegressionTrainer tests
│   └── test_external_datasets.py        # External loader and schema tests
├── utils/
│   ├── config.py                         # YAML namespace config loader
│   ├── logging.py                        # Console output formatter
│   └── tracker.py                        # CSV/JSON run recorder
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

**Requirements**: Python 3.12+

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```
numpy>=1.20.0
pyyaml>=6.0
pytest>=7.0.0
scikit-learn>=1.3.0
```

---

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: **31+ tests pass** in under 2 seconds.

---

## 🚀 Running Experiments

### Classical router baselines

```bash
PYTHONPATH=. python -m experiments.run --config experiments/configs/base_config.yaml
```

### Learned router baseline (Milestone 6)

```bash
PYTHONPATH=. python -m training.train
```

Runs the full 8-step pipeline:
1. Load 59-item benchmark
2. Convert to per-chunk ML training rows
3. Stratified 70/15/15 document-level split
4. Extract 7-dim feature vectors
5. Train `LogisticRegressionTrainer`
6. Evaluate: Precision, Recall, F1, AUC, NDCG@3/5, Recall@1/3/5
7. Save `data/training/{train,val,test}.jsonl`
8. Save `models/saved/lr_router.pkl`

---

## 📊 Dataset Formats

### Benchmark format (`data/datasets/raw/benchmark_data.json`)

59 synthetic legal QA items (NDAs, employment contracts, SaaS ToS, IP agreements…):

```json
{
  "document_id": "nda_01",
  "document_text": "...",
  "question": "What is the governing law?",
  "answer": "Delaware",
  "difficulty": "easy",
  "provenance": { "source_type": "synthetic", "source_name": "...", "license": "..." },
  "chunks": [
    {
      "id": 0, "text": "...", "start_position": 0, "end_position": 93,
      "token_count": 14, "is_relevant": true, "relevance_score": 3
    }
  ]
}
```

**Graded relevance** (0–3):
| Score | Meaning |
|---|---|
| 0 | Irrelevant |
| 1 | Supporting context |
| 2 | Useful — narrows answer |
| 3 | Essential — direct answer |

### ML training format (`data/training/train.jsonl`)

Per-chunk rows produced by `BenchmarkToMLConverter`:

```json
{
  "doc_id": "nda_01", "chunk_id": 1, "query": "What is the governing law?",
  "chunk_text": "...", "label": 1, "relevance_score": 3,
  "difficulty": "easy", "chunk_pos": 1, "total_chunks": 3, "chunk_tokens": 14
}
```

### External datasets

| Source | Format | Compatible? |
|---|---|---|
| **CUAD** | SQuAD-style span QA | ✅ Full |
| **LexGLUE / EURLEX** | Multi-label classification | ✅ Partial |
| **LexGLUE / ECtHR** | Binary / multi-label | ✅ Partial |
| **LexGLUE / UNFAIR-ToS** | Clause classification | ✅ Partial |
| **LexGLUE / SCOTUS** | Multi-class topic | ❌ Incompatible (no query/chunk structure) |
| **LexGLUE / CaseHOLD** | Multiple-choice | ❌ Incompatible |

---

## 📈 Current Baseline Results (59-item benchmark, threshold=0.4)

| Scorer | Precision | Recall | F1 | NDCG | Compression |
|---|---|---|---|---|---|
| Random | 50.0% | 54.2% | 48.3% | 0.49 | 53.1% |
| BM25 | 29.2% | 25.0% | 25.0% | 0.33 | 18.1% |
| TF-IDF | 75.0% | 70.8% | 66.1% | 0.73 | 64.9% |
| Trigram Embedding | 56.2% | 100.0% | 69.7% | 0.67 | 100.0% |
| **Learned LR Router** | **69.2%** | **75.0%** | **72.0%** | **0.85** | — |

---

## 🗺️ Milestone Roadmap

| Milestone | Status | Description |
|---|---|---|
| 1 | ✅ | Experimental framework and CLI |
| 2 | ✅ | Legal QA benchmark dataset |
| 3 | ✅ | Evaluation framework (precision, recall, NDCG) |
| 4.1 | ✅ | Embedding scorer calibration |
| 4.2 | ✅ | Baseline sweep and Pareto analysis |
| 5 | ✅ | Benchmark expansion (12 → 59 items) |
| 6 | ✅ | ML training pipeline + learned router baseline |
| 7 | ✅ | External dataset integration (CUAD, LexGLUE) |
| 8 | 🔜 | Large-scale corpus assembly + neural router training |
| 9 | 🔜 | Qwen/SmolLM fine-tuning and evaluation |
