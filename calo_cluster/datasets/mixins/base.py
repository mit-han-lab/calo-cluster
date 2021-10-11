from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from torch.utils.data.dataset import Dataset


class AbstractBaseDataset(ABC):
    @abstractmethod
    def _get_numpy(self, index: int) -> Dict[str, Any]:
        """Returns data for a given file index. Must include at least 'features' and 'coordinates'."""
        pass

class AbstractBaseDataModule(ABC):
    @abstractmethod
    def make_dataset_kwargs(self) -> Dict[str, Any]:
        """Makes kwargs for dataset from attributes of this datamodule."""
        pass

    @abstractmethod
    def make_dataset(self, files: List[Path], split: str) -> AbstractBaseDataset:
        pass