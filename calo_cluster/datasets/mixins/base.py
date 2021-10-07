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


@dataclass
class BaseDataset(AbstractBaseDataset, Dataset):
    """Base torch dataset.

    A subclass of this dataset needs to:
    1. override _get_numpy().
    2. override collate_fn if a different collate method is required for the dataset."""
    files: List[Path]
    voxel_size: float

    def __post_init__(self):
        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return_dict = self._get_numpy(index)
        self._validate(return_dict)
        return return_dict

    def _validate(return_dict):
        if 'features' not in return_dict or 'coordinates' not in return_dict:
            raise RuntimeError(
                'return_dict must contain "features" and "coordinates"!')

    @property
    def collate_fn(self) -> Callable[[List[Any]], Any]:
        """Returns a function that collates data into batches for the dataloader."""
        return None
