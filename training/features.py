"""
training/features.py
--------------------
Feature extraction for the learned router baseline.

Produces a fixed-length 7-dimensional numpy feature vector for each
(query, chunk) pair. All features are normalised to [0.0, 1.0].

Feature index map
-----------------
  0  term_overlap_ratio      — lexical overlap between query and chunk terms
  1  tfidf_cosine_sim        — TF-IDF cosine similarity (TFIDFScorer)
  2  bm25_score_norm         — BM25 score (BM25Scorer), normalised per row
  3  trigram_embedding_sim   — char-trigram cosine similarity (EmbeddingScorer)
  4  chunk_position_norm     — chunk position in document (0 = first)
  5  chunk_token_norm        — chunk token count, capped at 512
  6  query_length_norm       — query word count, capped at 50
"""
from __future__ import annotations

from typing import List

import numpy as np

from baselines.bm25 import BM25Scorer
from baselines.embedding import EmbeddingScorer
from baselines.tfidf import TFIDFScorer

_TFIDF_SCORER = TFIDFScorer()
_BM25_SCORER = BM25Scorer()
_EMBEDDING_SCORER = EmbeddingScorer()

FEATURE_DIM = 7
FEATURE_NAMES = [
    "term_overlap_ratio",
    "tfidf_cosine_sim",
    "bm25_score_norm",
    "trigram_embedding_sim",
    "chunk_position_norm",
    "chunk_token_norm",
    "query_length_norm",
]


def _term_overlap_ratio(query: str, chunk_text: str) -> float:
    """Fraction of unique query terms that appear in the chunk."""
    q_terms = set(query.lower().split())
    c_terms = set(chunk_text.lower().split())
    if not q_terms:
        return 0.0
    return len(q_terms & c_terms) / len(q_terms)


class FeatureExtractor:
    """
    Extracts a 7-dimensional feature vector for a single (query, chunk) pair.

    Usage
    -----
    extractor = FeatureExtractor()

    # Single pair
    vec = extractor.extract(query, chunk_text, chunk_pos=0, total_chunks=4, chunk_tokens=35)

    # Batch (full document context — preferred for accurate BM25/TF-IDF scores)
    matrix = extractor.extract_batch(query, chunks_text, chunk_tokens_list)
    """

    # -------------------------------------------------------------------------
    # Single-pair extraction (uses single-item scorer calls)
    # -------------------------------------------------------------------------
    def extract(
        self,
        query: str,
        chunk_text: str,
        chunk_pos: int,
        total_chunks: int,
        chunk_tokens: int,
    ) -> np.ndarray:
        """Return a 7-float feature vector for one (query, chunk) pair.

        Note: single-pair calls wrap the chunk in a one-element list for the
        scorers, so BM25/TF-IDF IDF distributions are computed over only this
        chunk (degenerate case).  For more accurate features use extract_batch.
        """
        return self.extract_batch(
            query=query,
            chunks=[chunk_text],
            chunk_tokens_list=[chunk_tokens],
        )[chunk_pos if chunk_pos == 0 else 0]

    # -------------------------------------------------------------------------
    # Batch extraction — preferred for a full document's chunks
    # -------------------------------------------------------------------------
    def extract_batch(
        self,
        query: str,
        chunks: List[str],
        chunk_tokens_list: List[int],
    ) -> np.ndarray:
        """
        Return an (N, 7) feature matrix for all chunks in a document.

        Args:
            query:            The question/query string.
            chunks:           List[str] of chunk texts for this document.
            chunk_tokens_list: Token count per chunk (parallel to `chunks`).

        Returns:
            np.ndarray of shape (N, 7), dtype float32.
        """
        n = len(chunks)
        if n == 0:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)

        total_chunks = n
        q_len_norm = min(len(query.split()) / 50.0, 1.0)

        # Scorer calls (return List[float] of length n)
        tfidf_scores  = _TFIDF_SCORER.score(query, chunks)
        bm25_scores   = _BM25_SCORER.score(query, chunks)
        trigram_scores = _EMBEDDING_SCORER.score(query, chunks)

        # Normalise BM25 scores within this document [0, 1]
        bm25_arr = np.array(bm25_scores, dtype=np.float32)
        bm25_max = bm25_arr.max()
        if bm25_max > 0:
            bm25_arr = bm25_arr / bm25_max

        vectors = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        for i, chunk_text in enumerate(chunks):
            vectors[i, 0] = _term_overlap_ratio(query, chunk_text)
            vectors[i, 1] = float(np.clip(tfidf_scores[i],  0.0, 1.0))
            vectors[i, 2] = float(np.clip(bm25_arr[i],      0.0, 1.0))
            vectors[i, 3] = float(np.clip(trigram_scores[i], 0.0, 1.0))
            vectors[i, 4] = i / max(total_chunks - 1, 1)
            vectors[i, 5] = min(chunk_tokens_list[i] / 512.0, 1.0)
            vectors[i, 6] = q_len_norm

        return vectors
