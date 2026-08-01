"""
data/datasets/external_statistics.py
--------------------------------------
Generates markdown statistics reports for external router datasets (CUAD, LexGLUE, etc.)
and writes output to reports/external_dataset_statistics.md.

Usage
-----
    PYTHONPATH=. python -m data.datasets.external_statistics
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional

from training.ml_dataset import MLTrainingExample

REPORT_PATH = Path("docs/external_dataset_statistics.md")


def generate_external_report(
    examples: List[MLTrainingExample], source_name: str = "External Dataset"
) -> str:
    """
    Generate a markdown statistics report string for a collection of MLTrainingExample items.
    """
    total = len(examples)
    unique_docs = len({ex.doc_id for ex in examples})
    unique_queries = len({ex.query for ex in examples})

    label_counts = Counter(ex.label for ex in examples)
    score_counts = Counter(ex.relevance_score for ex in examples)
    diff_counts = Counter(ex.difficulty for ex in examples)

    token_counts = [ex.chunk_tokens for ex in examples]
    avg_tokens = sum(token_counts) / total if total > 0 else 0.0

    lines = [
        f"# External Dataset Statistics Report — {source_name}",
        "",
        f"**Total ML training rows**: {total}",
        f"**Unique documents**: {unique_docs}",
        f"**Unique queries**: {unique_queries}",
        f"**Average tokens per chunk**: {avg_tokens:.2f}",
        "",
        "## Binary Label Distribution",
        f"- Irrelevant (0): {label_counts[0]} ({label_counts[0] / total * 100:.1f}%)" if total else "- Irrelevant (0): 0",
        f"- Relevant (1): {label_counts[1]} ({label_counts[1] / total * 100:.1f}%)" if total else "- Relevant (1): 0",
        "",
        "## Relevance Score Distribution (0–3)",
    ]

    for score in sorted(score_counts.keys()):
        cnt = score_counts[score]
        pct = (cnt / total * 100) if total > 0 else 0.0
        lines.append(f"- Score {score}: {cnt} ({pct:.1f}%)")

    lines.extend([
        "",
        "## Difficulty Level Distribution",
    ])
    for diff, cnt in sorted(diff_counts.items()):
        lines.append(f"- {diff}: {cnt}")

    lines.extend([
        "",
        "## Dataset Compatibility & Integration Notes",
        "- External datasets are mapped into `MLTrainingExample` dataclasses.",
        "- CUAD maps clause annotations to binary 0/1 and graded relevance 0 or 3.",
        "- LexGLUE compatible tasks (EURLEX, ECtHR, UNFAIR-ToS) map task labels to binary 0/1.",
        "",
    ])

    return "\n".join(lines)


def main(examples: Optional[List[MLTrainingExample]] = None, source_name: str = "Unified External Sources"):
    if examples is None:
        # Create a synthetic demo dataset if raw files are not downloaded
        from training.ml_dataset import MLTrainingExample
        examples = [
            MLTrainingExample(
                doc_id="cuad_demo_01",
                chunk_id=0,
                query="What is the governing law?",
                chunk_text="This agreement is governed by Delaware law.",
                label=1,
                relevance_score=3,
                difficulty="external",
                chunk_pos=0,
                total_chunks=1,
                chunk_tokens=7,
            ),
            MLTrainingExample(
                doc_id="lexglue_demo_01",
                chunk_id=0,
                query="Does this clause represent an unfair term?",
                chunk_text="The company reserves the right to terminate service at any time.",
                label=1,
                relevance_score=1,
                difficulty="external",
                chunk_pos=0,
                total_chunks=1,
                chunk_tokens=11,
            ),
        ]
        source_name = "Synthetic Demonstration Examples"

    md = generate_external_report(examples, source_name=source_name)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"External dataset statistics written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
