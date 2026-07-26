from .loaders.document_loader import LegalDocumentLoader
from .preprocess.splitter import DocumentSplitter
from .datasets.dataset_interface import BaseDataset, BaseDatasetLoader
from .datasets.benchmark_schema import Chunk, BenchmarkQAItem
from .datasets.benchmark_loader import BenchmarkLegalDataset, BenchmarkDatasetLoader

__all__ = [
    "LegalDocumentLoader",
    "DocumentSplitter",
    "BaseDataset",
    "BaseDatasetLoader",
    "Chunk",
    "BenchmarkQAItem",
    "BenchmarkLegalDataset",
    "BenchmarkDatasetLoader"
]
