from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class AbstractBaseDataset(ABC):
    @abstractmethod
    def _get_numpy(self, index: int) -> Dict[str, Any]:
        """Returns data for a given file index. Must include at least 'features' and 'coordinates'."""

class AbstractBaseDataModule(ABC):
    @abstractmethod
    def make_dataset_kwargs(self) -> Dict[str, Any]:
        """Makes kwargs for dataset from attributes of this datamodule."""

    @abstractmethod
    def make_dataset(self, files: List[Path], split: str) -> AbstractBaseDataset:
        pass