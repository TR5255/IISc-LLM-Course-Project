# dataset_statistics.py
"""Utility to compute high‑level statistics for the legal QA benchmark.
It loads the benchmark via ``BenchmarkDatasetLoader`` and writes a markdown
report to ``reports/dataset_statistics.md``.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

# Import the loader from the project (relative import works when PYTHONPATH=.)
from data.datasets.benchmark_loader import BenchmarkDatasetLoader

DATA_PATH = Path("data/datasets/raw/benchmark_data.json")
REPORT_PATH = Path("reports/dataset_statistics.md")

def load_dataset():
    loader = BenchmarkDatasetLoader(str(DATA_PATH))
    return loader.load()

def compute_statistics(dataset):
    num_documents = len(dataset)
    num_qa = len(dataset)  # one QA per document in current schema
    total_chunks = sum(len(item.chunks) for item in dataset)

    # Document distribution by category (infer from document_id prefix)
    cat_counts = Counter()
    for item in dataset:
        prefix = item.document_id.split('_')[0]
        cat_counts[prefix] += 1

    # Chunk statistics
    chunks_per_doc = [len(item.chunks) for item in dataset]
    avg_chunks = sum(chunks_per_doc) / num_documents if num_documents else 0
    token_counts = []
    for item in dataset:
        for c in item.chunks:
            token_counts.append(c.token_count)
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    # Relevance label distribution
    relevance_counter = Counter()
    for item in dataset:
        for c in item.chunks:
            relevance_counter[c.relevance_score] += 1

    # Difficulty distribution
    difficulty_counter = Counter(item.difficulty for item in dataset)

    return {
        "num_documents": num_documents,
        "num_qa": num_qa,
        "total_chunks": total_chunks,
        "cat_counts": dict(cat_counts),
        "avg_chunks": avg_chunks,
        "avg_tokens": avg_tokens,
        "relevance_counter": dict(relevance_counter),
        "difficulty_counter": dict(difficulty_counter),
    }

def render_markdown(stats):
    lines = []
    lines.append("# Dataset Statistics Report")
    lines.append("")
    lines.append(f"**Documents**: {stats['num_documents']}")
    lines.append(f"**QA pairs**: {stats['num_qa']}")
    lines.append(f"**Total chunks**: {stats['total_chunks']}")
    lines.append("")
    lines.append("## Document Distribution by Category")
    for cat, cnt in sorted(stats['cat_counts'].items()):
        lines.append(f"- {cat}: {cnt}")
    lines.append("")
    lines.append("## Chunk Statistics")
    lines.append(f"- Average chunks per document: {stats['avg_chunks']:.2f}")
    lines.append(f"- Average token count per chunk: {stats['avg_tokens']:.2f}")
    lines.append("")
    lines.append("## Relevance Score Distribution")
    for score in sorted(stats['relevance_counter'].keys()):
        lines.append(f"- Score {score}: {stats['relevance_counter'][score]}")
    lines.append("")
    lines.append("## Difficulty Distribution")
    for diff, cnt in sorted(stats['difficulty_counter'].items()):
        lines.append(f"- {diff}: {cnt}")
    lines.append("")
    return "\n".join(lines)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for legal QA benchmarks.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DATA_PATH),
        help="Path to JSON dataset source file (default: data/datasets/raw/benchmark_data.json)",
    )
    args = parser.parse_args()

    loader = BenchmarkDatasetLoader(args.source)
    dataset = loader.load()
    stats = compute_statistics(dataset)
    md = render_markdown(stats)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"Dataset statistics for '{args.source}' written to {REPORT_PATH}")

if __name__ == "__main__":
    main()

