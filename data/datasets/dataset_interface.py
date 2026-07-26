from abc import ABC, abstractmethod
from typing import List, Any

class BaseDataset(ABC):
    """Abstract Base Class for all datasets (mock, synthetic, or real)."""
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass


class BaseDatasetLoader(ABC):
    """Abstract Base Class for loading datasets from files or remote sources."""
    
    @abstractmethod
    def load(self, *args, **kwargs) -> BaseDataset:
        """Loads and returns a BaseDataset instance."""
        pass
