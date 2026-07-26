import json
import os
from typing import List
from data.datasets.dataset_interface import BaseDataset, BaseDatasetLoader
from data.datasets.benchmark_schema import BenchmarkQAItem

class BenchmarkLegalDataset(BaseDataset):
    """Container holding parsed legal QA benchmark items."""
    def __init__(self, items: List[BenchmarkQAItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BenchmarkQAItem:
        return self.items[idx]


class BenchmarkDatasetLoader(BaseDatasetLoader):
    """Loads benchmark items from the compiled JSON database."""
    def __init__(self, json_path: str = "data/datasets/raw/benchmark_data.json"):
        # Resolve path relative to maddy_project root if needed
        self.json_path = json_path

    def load(self, *args, **kwargs) -> BenchmarkLegalDataset:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Benchmark data file not found: {self.json_path}")
            
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Root JSON element must be a list of items.")
            
        items = [BenchmarkQAItem.from_dict(item) for item in data]
        return BenchmarkLegalDataset(items)
