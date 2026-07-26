"""
data/external/cuad/cuad_loader.py
----------------------------------
Loads the CUAD v1 dataset (SQuAD-format JSON) and converts it into
MLTrainingExample rows for the Smart AI Router training pipeline.

CUAD schema → MLTrainingExample mapping
----------------------------------------
  CUAD field               MLTrainingExample field
  ─────────────────────────────────────────────────
  title                    doc_id
  context (chunked)        chunk_text
  question                 query
  answers.answer_start/text → label=1, relevance_score=3
  no answer span           → label=0, relevance_score=0
  —                        difficulty = "external"

Graded relevance note
----------------------
CUAD only distinguishes presence/absence of a clause span.  Intermediate
relevance grades (1 and 2) are not producible from CUAD annotations alone.
All positive labels are assigned relevance_score=3 (essential).

Usage
-----
    from data.external.cuad.cuad_loader import CUADLoader
    examples = CUADLoader().load("data/external/cuad/raw/CUAD_v1.json")
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from training.ml_dataset import MLTrainingExample

logger = logging.getLogger(__name__)

# Approximate token count: split on whitespace.
# A more sophisticated tokeniser can be swapped in later.
CHUNK_TOKEN_LIMIT = 512


def _whitespace_tokens(text: str) -> List[str]:
    return text.split()


def _chunk_context(context: str) -> List[Tuple[str, int, int]]:
    """
    Split a CUAD paragraph context into non-overlapping chunks of up to
    CHUNK_TOKEN_LIMIT whitespace tokens.

    Returns list of (chunk_text, char_start, char_end).
    """
    words = _whitespace_tokens(context)
    chunks: List[Tuple[str, int, int]] = []
    char_ptr = 0

    i = 0
    while i < len(words):
        window = words[i : i + CHUNK_TOKEN_LIMIT]
        chunk_text = " ".join(window)
        char_start = context.find(chunk_text, char_ptr)
        if char_start == -1:
            # Fallback: advance char pointer by approximate length
            char_start = char_ptr
        char_end = char_start + len(chunk_text)
        chunks.append((chunk_text, char_start, char_end))
        char_ptr = char_end
        i += CHUNK_TOKEN_LIMIT

    return chunks


def _spans_overlap(
    chunk_start: int, chunk_end: int, ans_start: int, ans_end: int
) -> bool:
    """Return True if the answer span overlaps with the chunk character range."""
    return chunk_start < ans_end and ans_start < chunk_end


class CUADLoader:
    """
    Loads CUAD v1 JSON and converts to List[MLTrainingExample].

    Parameters
    ----------
    max_contracts : int or None
        If set, only load the first N contracts (useful for fast testing).
    """

    def __init__(self, max_contracts: Optional[int] = None):
        self.max_contracts = max_contracts

    def load(self, json_path: str) -> List[MLTrainingExample]:
        """
        Load CUAD from *json_path* and return MLTrainingExample list.

        Raises FileNotFoundError if the path does not exist.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"CUAD data not found at: {json_path}\n"
                "Run: bash data/external/cuad/download_cuad.sh"
            )

        with open(json_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)

        contracts = raw.get("data", [])
        if self.max_contracts is not None:
            contracts = contracts[: self.max_contracts]

        examples: List[MLTrainingExample] = []
        for contract in contracts:
            examples.extend(self._process_contract(contract))

        logger.info("CUADLoader: produced %d examples from %d contracts.",
                    len(examples), len(contracts))
        return examples

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _process_contract(self, contract: Dict[str, Any]) -> List[MLTrainingExample]:
        title = contract.get("title", "unknown")
        # CUAD wraps each contract in a single-element paragraphs list
        paragraphs = contract.get("paragraphs", [])
        examples: List[MLTrainingExample] = []

        for para in paragraphs:
            context: str = para.get("context", "")
            qas: List[Dict] = para.get("qas", [])
            chunks = _chunk_context(context)

            for qa in qas:
                query: str = qa.get("question", "")
                answers: List[Dict] = qa.get("answers", [])
                # Build set of (ans_start, ans_end) character spans
                ans_spans: List[Tuple[int, int]] = [
                    (a["answer_start"], a["answer_start"] + len(a["text"]))
                    for a in answers
                    if "answer_start" in a and "text" in a
                ]

                for pos, (chunk_text, c_start, c_end) in enumerate(chunks):
                    # A chunk is relevant if any answer span overlaps it
                    is_relevant = any(
                        _spans_overlap(c_start, c_end, a_s, a_e)
                        for a_s, a_e in ans_spans
                    )
                    doc_id = f"cuad__{title.replace(' ', '_')}"
                    examples.append(
                        MLTrainingExample(
                            doc_id=doc_id,
                            chunk_id=pos,
                            query=query,
                            chunk_text=chunk_text,
                            label=int(is_relevant),
                            relevance_score=3 if is_relevant else 0,
                            difficulty="external",
                            chunk_pos=pos,
                            total_chunks=len(chunks),
                            chunk_tokens=len(_whitespace_tokens(chunk_text)),
                        )
                    )
        return examples


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/external/cuad/raw/CUAD_v1.json"
    loader = CUADLoader()
    exs = loader.load(path)
    pos = sum(e.label for e in exs)
    print(f"Loaded {len(exs)} examples from CUAD ({pos} positive / {len(exs)-pos} negative)")
