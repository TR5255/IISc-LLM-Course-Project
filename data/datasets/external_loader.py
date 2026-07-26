"""
data/datasets/external_loader.py
---------------------------------
Unified interface for loading and combining external dataset sources
(CUAD, LexGLUE, etc.) into MLTrainingExample collections for the Smart AI Router.

Usage
-----
    from data.datasets.external_loader import (
        CUADExternalLoader,
        LexGLUEExternalLoader,
        UnifiedRouterDataset,
    )

    cuad_loader = CUADExternalLoader("data/external/cuad/raw/CUAD_v1.json")
    lexglue_loader = LexGLUEExternalLoader(task_name="eurlex", use_hf=True)

    unified = UnifiedRouterDataset([cuad_loader, lexglue_loader])
    examples = unified.load_all()
    stats = unified.summary()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import logging
from typing import Any, Dict, List, Optional

from training.ml_dataset import MLTrainingExample
from data.external.cuad.cuad_loader import CUADLoader
from data.external.lexglue.lexglue_loader import LexGLUELoader

logger = logging.getLogger(__name__)


class ExternalDatasetLoader(ABC):
    """Abstract base class for external dataset loaders."""

    @abstractmethod
    def load(self) -> List[MLTrainingExample]:
        """Load dataset and return list of MLTrainingExample items."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Return human-readable identifier for this dataset source."""
        ...


class CUADExternalLoader(ExternalDatasetLoader):
    """Adapter for CUAD dataset loader."""

    def __init__(self, json_path: str = "data/external/cuad/raw/CUAD_v1.json", max_contracts: Optional[int] = None):
        self.json_path = json_path
        self.max_contracts = max_contracts
        self._loader = CUADLoader(max_contracts=max_contracts)

    def load(self) -> List[MLTrainingExample]:
        return self._loader.load(self.json_path)

    def source_name(self) -> str:
        return "cuad"


class LexGLUEExternalLoader(ExternalDatasetLoader):
    """Adapter for LexGLUE dataset loader."""

    def __init__(
        self,
        task_name: str = "eurlex",
        use_hf: bool = False,
        data_dir: Optional[str] = None,
        split: str = "train",
        max_examples: Optional[int] = None,
    ):
        self.task_name = task_name
        self.use_hf = use_hf
        self.data_dir = data_dir
        self.split = split
        self.max_examples = max_examples
        self._loader = LexGLUELoader(use_hf=use_hf)

    def load(self) -> List[MLTrainingExample]:
        return self._loader.load(
            task_name=self.task_name,
            data_dir=self.data_dir,
            split=self.split,
            max_examples=self.max_examples,
        )

    def source_name(self) -> str:
        return f"lexglue_{self.task_name}"


class UnifiedRouterDataset:
    """
    Aggregates multiple ExternalDatasetLoader instances, loads examples across
    sources, and provides summary metrics.
    """

    def __init__(self, loaders: Optional[List[ExternalDatasetLoader]] = None):
        self.loaders: List[ExternalDatasetLoader] = loaders or []

    def add_loader(self, loader: ExternalDatasetLoader) -> None:
        """Add a dataset loader source."""
        self.loaders.append(loader)

    def load_all(self) -> List[MLTrainingExample]:
        """Load examples from all configured loaders."""
        all_examples: List[MLTrainingExample] = []
        for loader in self.loaders:
            try:
                examples = loader.load()
                all_examples.extend(examples)
                logger.info("UnifiedRouterDataset: loaded %d examples from '%s'.",
                            len(examples), loader.source_name())
            except FileNotFoundError as err:
                logger.warning("UnifiedRouterDataset: skipping '%s' (%s).",
                               loader.source_name(), err)
            except Exception as err:
                logger.error("UnifiedRouterDataset: error loading '%s': %s",
                             loader.source_name(), err)
        return all_examples

    def summary(self, examples: Optional[List[MLTrainingExample]] = None) -> Dict[str, Any]:
        """
        Compute summary statistics for a given example list (or loads all if None).
        """
        if examples is None:
            examples = self.load_all()

        total_count = len(examples)
        label_dist = Counter(ex.label for ex in examples)
        score_dist = Counter(ex.relevance_score for ex in examples)
        doc_ids = {ex.doc_id for ex in examples}
        
        # Source breakdown from doc_id prefix
        sources = Counter()
        for ex in examples:
            prefix = ex.doc_id.split("__")[0] if "__" in ex.doc_id else "unknown"
            sources[prefix] += 1

        avg_tokens = (
            sum(ex.chunk_tokens for ex in examples) / total_count if total_count > 0 else 0.0
        )

        return {
            "total_examples": total_count,
            "unique_documents": len(doc_ids),
            "label_distribution": dict(label_dist),
            "relevance_score_distribution": dict(score_dist),
            "source_counts": dict(sources),
            "avg_chunk_tokens": round(avg_tokens, 2),
        }
