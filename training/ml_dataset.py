"""
training/ml_dataset.py
----------------------
Converts the BenchmarkLegalDataset into flat per-chunk ML training examples.

Each BenchmarkQAItem produces N MLTrainingExample rows — one per chunk —
carrying both the binary label (is_relevant) and the graded relevance_score
(0-3) so that ranking metrics (NDCG, Recall@K) can be computed later without
any data loss.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class MLTrainingExample:
    """A single per-chunk training row derived from a BenchmarkQAItem."""
    # Query / chunk identity
    doc_id: str
    chunk_id: int
    query: str
    chunk_text: str

    # Labels — both kept; binary label for classification, graded for ranking
    label: int           # 0 or 1  (from is_relevant)
    relevance_score: int # 0-3     (graded relevance — never discarded)

    # Structural context (used by FeatureExtractor)
    difficulty: str
    chunk_pos: int       # 0-indexed position in document
    total_chunks: int    # total chunk count for this document
    chunk_tokens: int    # token count of this chunk


class BenchmarkToMLConverter:
    """Flattens a BenchmarkLegalDataset into a list of MLTrainingExample rows."""

    def convert(self, dataset) -> List[MLTrainingExample]:
        """
        Convert every item → list of per-chunk rows.

        Args:
            dataset: BenchmarkLegalDataset (iterable of BenchmarkQAItem)

        Returns:
            List[MLTrainingExample] — one per chunk across all items.
        """
        examples: List[MLTrainingExample] = []
        for item in dataset.items:
            total_chunks = len(item.chunks)
            for pos, chunk in enumerate(item.chunks):
                examples.append(
                    MLTrainingExample(
                        doc_id=item.document_id,
                        chunk_id=chunk.id,
                        query=item.question,
                        chunk_text=chunk.text,
                        label=int(chunk.is_relevant),
                        relevance_score=chunk.relevance_score,
                        difficulty=item.difficulty,
                        chunk_pos=pos,
                        total_chunks=total_chunks,
                        chunk_tokens=chunk.token_count,
                    )
                )
        return examples


# ---------------------------------------------------------------------------
# JSONL serialisation helpers
# ---------------------------------------------------------------------------

def save_jsonl(examples: List[MLTrainingExample], path: str) -> None:
    """Persist a list of MLTrainingExample to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex)) + "\n")


def load_jsonl(path: str) -> List[MLTrainingExample]:
    """Load MLTrainingExample rows from a JSONL file."""
    examples: List[MLTrainingExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(MLTrainingExample(**json.loads(line)))
    return examples
