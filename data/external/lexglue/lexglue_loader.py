"""
data/external/lexglue/lexglue_loader.py
----------------------------------------
Loads LexGLUE task data and converts compatible tasks into MLTrainingExample
rows for the Smart AI Router training pipeline.

LexGLUE task compatibility
---------------------------
  Task         | Type               | Compatible | Mapping
  ─────────────────────────────────────────────────────────────────────────
  EURLEX       | Multi-label class. | ✅ Partial | doc→chunks; label presence
  ECtHR (A)    | Binary class.      | ✅ Partial | doc→chunks; binary label
  ECtHR (B)    | Multi-label class. | ✅ Partial | same as EURLEX
  UNFAIR-ToS   | Multi-label class. | ✅ Partial | clause relevance
  SCOTUS       | Multi-class topic  | ❌ Skip    | no query/chunk structure
  LEDGAR       | Multi-class class. | ❌ Skip    | provision classification only
  CaseHOLD     | Multiple-choice    | ❌ Skip    | different retrieval structure

Compatible tasks produce:
  - query  = task description string (the "routing question")
  - chunk_text = document passage chunk (~512 tokens)
  - label  = 1 if the document class label is relevant, else 0
  - relevance_score = 1 (partial — LexGLUE has no graded relevance)

Incompatible tasks log a warning and return [].

Usage (library)
---------------
    from data.external.lexglue.lexglue_loader import LexGLUELoader
    examples = LexGLUELoader().load("eurlex", data_dir="data/external/lexglue/raw")

Usage (HuggingFace datasets — optional)
-----------------------------------------
    # Install: pip install datasets
    from data.external.lexglue.lexglue_loader import LexGLUELoader
    examples = LexGLUELoader(use_hf=True).load("eurlex")
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from training.ml_dataset import MLTrainingExample

logger = logging.getLogger(__name__)

CHUNK_TOKEN_LIMIT = 512

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

# Maps task name → (compatible, query_template, label_type)
_TASK_META: Dict[str, Dict[str, Any]] = {
    "eurlex": {
        "compatible": True,
        "query": "Is this document relevant to the following EU legal concept?",
        "label_type": "multilabel",
    },
    "ecthr_a": {
        "compatible": True,
        "query": "Does this court ruling involve a violation of this article?",
        "label_type": "binary",
    },
    "ecthr_b": {
        "compatible": True,
        "query": "Is this article alleged to have been violated in this case?",
        "label_type": "multilabel",
    },
    "unfair_tos": {
        "compatible": True,
        "query": "Does this clause represent an unfair term of service?",
        "label_type": "multilabel",
    },
    "scotus": {
        "compatible": False,
        "reason": "SCOTUS is a multi-class topic classification task with no query/chunk retrieval structure.",
    },
    "ledgar": {
        "compatible": False,
        "reason": "LEDGAR classifies legal provision types; it does not have a query/chunk routing structure.",
    },
    "case_hold": {
        "compatible": False,
        "reason": "CaseHOLD is a multiple-choice holding identification task; incompatible with chunk-relevance routing.",
    },
}

COMPATIBLE_TASKS = {k for k, v in _TASK_META.items() if v.get("compatible")}
INCOMPATIBLE_TASKS = {k for k, v in _TASK_META.items() if not v.get("compatible")}


def _chunk_text(text: str) -> List[str]:
    """Split text into chunks of ~CHUNK_TOKEN_LIMIT whitespace tokens."""
    words = text.split()
    return [
        " ".join(words[i : i + CHUNK_TOKEN_LIMIT])
        for i in range(0, len(words), CHUNK_TOKEN_LIMIT)
    ]


class LexGLUELoader:
    """
    Loads a single LexGLUE task and converts to List[MLTrainingExample].

    Parameters
    ----------
    use_hf : bool
        If True, attempt to load via ``datasets.load_dataset("lex_glue", task_name)``.
        If False (default), read from local JSONL files in *data_dir*.
    """

    def __init__(self, use_hf: bool = False):
        self.use_hf = use_hf

    def load(
        self,
        task_name: str,
        data_dir: Optional[str] = None,
        split: str = "train",
        max_examples: Optional[int] = None,
    ) -> List[MLTrainingExample]:
        """
        Load *task_name* from LexGLUE.

        Returns [] and logs a warning for incompatible tasks.
        """
        task_key = task_name.lower()

        if task_key not in _TASK_META:
            logger.warning("LexGLUELoader: unknown task '%s'. Returning [].", task_name)
            return []

        meta = _TASK_META[task_key]
        if not meta.get("compatible"):
            logger.warning(
                "LexGLUELoader: task '%s' is not compatible with router training. "
                "Reason: %s  Returning [].",
                task_name, meta.get("reason", ""),
            )
            return []

        if self.use_hf:
            return self._load_hf(task_key, meta, split, max_examples)
        else:
            return self._load_local(task_key, meta, data_dir or "data/external/lexglue/raw",
                                    split, max_examples)

    # -------------------------------------------------------------------------
    # HuggingFace path
    # -------------------------------------------------------------------------
    def _load_hf(
        self, task_key: str, meta: Dict, split: str, max_examples: Optional[int]
    ) -> List[MLTrainingExample]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "Install 'datasets' to use use_hf=True: pip install datasets"
            )

        ds = load_dataset("lex_glue", task_key, split=split, trust_remote_code=True)
        records = list(ds)
        if max_examples:
            records = records[:max_examples]
        return self._convert(task_key, meta, records)

    # -------------------------------------------------------------------------
    # Local JSONL path
    # -------------------------------------------------------------------------
    def _load_local(
        self,
        task_key: str,
        meta: Dict,
        data_dir: str,
        split: str,
        max_examples: Optional[int],
    ) -> List[MLTrainingExample]:
        file_path = os.path.join(data_dir, task_key, f"{split}.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"LexGLUE local data not found: {file_path}\n"
                "Download via: pip install datasets && python -c \""
                f"from datasets import load_dataset; load_dataset('lex_glue', '{task_key}')\""
            )
        records = []
        with open(file_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if max_examples:
            records = records[:max_examples]
        return self._convert(task_key, meta, records)

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------
    def _convert(
        self, task_key: str, meta: Dict, records: List[Dict]
    ) -> List[MLTrainingExample]:
        query = meta["query"]
        label_type = meta["label_type"]
        examples: List[MLTrainingExample] = []

        for idx, record in enumerate(records):
            text: str = record.get("text", "")
            raw_label = record.get("label", 0)

            # Normalise label to binary
            if label_type == "multilabel":
                # multilabel: list of ints; document is relevant if any label > 0
                if isinstance(raw_label, list):
                    is_relevant = any(l > 0 for l in raw_label)
                else:
                    is_relevant = bool(raw_label)
            else:
                is_relevant = bool(raw_label)

            chunks = _chunk_text(text)
            total = len(chunks)
            doc_id = f"lexglue__{task_key}_{idx}"

            for pos, chunk_text in enumerate(chunks):
                examples.append(
                    MLTrainingExample(
                        doc_id=doc_id,
                        chunk_id=pos,
                        query=query,
                        chunk_text=chunk_text,
                        label=int(is_relevant),
                        # LexGLUE has no graded relevance; use 1 for partial
                        relevance_score=1 if is_relevant else 0,
                        difficulty="external",
                        chunk_pos=pos,
                        total_chunks=total,
                        chunk_tokens=len(chunk_text.split()),
                    )
                )
        logger.info("LexGLUELoader[%s]: produced %d examples.", task_key, len(examples))
        return examples
