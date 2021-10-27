import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import hydra
import numpy as np
import pytorch_lightning as pl
from calo_cluster.datasets.mixins.base import (AbstractBaseDataModule,
                                               AbstractBaseDataset)
from hydra import compose, initialize_config_dir
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm


@dataclass
class BaseDataset(AbstractBaseDataset, Dataset):
    """Base torch dataset that assumes 1 event per file.

    A subclass of this dataset needs to:
    1. override _get_numpy().
    2. override collate_fn if a different collate method is required for the dataset."""
    files: List[Path]

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


@dataclass
class BaseDataModule(AbstractBaseDataModule, pl.LightningDataModule):
    """The base pytorch-lightning data module that handles common data loading tasks.


    This module assumes that the data is organized into a set of files, with one event per file.
    When creating a base class, make sure to override make_dataset appropriately.

    Parameters:
    seed -- a seed used by the RNGs
    num_epochs -- the number of epochs
    batch_size -- the batch size

    num_workers -- the number of CPU processes to use for data workers.

    event_frac -- the fraction of total data to use
    train_frac -- the fraction of train data to use
    test_frac -- the fraction of test data to use

    cluster_ignore_labels -- the semantic labels that should be ignored when clustering (needs to be supported by clusterer) and in embed criterion (needs to be supported by embed criterion)
    semantic_ignore_label -- the semantic label that should be ignored in semantic segmentation criterion (needs to be supported by semantic criterion)

    batch_dim -- the dimension that contains batch information, if sparse=False. If sparse=True, the batch should be stored in the last dimension of the coordinates.

    num_classes -- the number of semantic classes
    num_features -- the number of features used as input to the ML model

    data_dir -- the base data directory
    """

    seed: int
    num_epochs: int
    batch_size: int

    num_workers: int

    event_frac: float
    train_frac: float
    test_frac: float

    cluster_ignore_labels: List[int]
    semantic_ignore_label: Union[int, None]

    batch_dim: int

    num_classes: int
    num_features: int

    data_dir: str

    @property
    def files(self) -> List[Path]:
        if self._files is None:
            self._files = []
            self._files.extend(
                sorted(self.data_dir.glob('*')))
        return self._files

    def __post_init__(self):
        super().__init__()

        self._validate_fracs()
        self._files = None
        self.data_dir = Path(self.data_dir)

    def _validate_fracs(self):
        fracs = [self.event_frac, self.train_frac, self.test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert self.train_frac + self.test_frac <= 1.0

    def train_val_test_split(self) -> Tuple[Union[List[Path], None], Union[List[Path], None], Union[List[Path], None]]:
        """Returns train, val, and test file lists

        Assumes that self.files is defined and there is no preset split in the dataset.
        If the dataset already has train/val/test files defined, override this function
        and return them."""
        files = shuffle(self.files, random_state=42)
        num_files = int(self.event_frac * len(files))
        files = files[:num_files]
        num_train_files = int(self.train_frac * num_files)
        num_test_files = int(self.test_frac * num_files)

        train_files = files[:num_train_files]
        val_files = files[num_train_files:-num_test_files]
        test_files = files[-num_test_files:]

        return train_files, val_files, test_files

    def setup(self, stage: str = None) -> None:
        train_files, val_files, test_files = self.train_val_test_split()

        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = self.make_dataset(train_files, split='train')
            self.val_dataset = self.make_dataset(val_files, split='val')
        if stage == 'test' or stage is None:
            self.test_dataset = self.make_dataset(test_files, split='test')

    def dataloader(self, dataset: BaseDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    def make_dataset_kwargs(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, overrides: List[str] = []):
        config_dir = Path(__file__).parent.parent.parent / 'configs'
        overrides.append('train.batch_size=1')
        overrides = cls.fix_overrides(overrides)
        with initialize_config_dir(config_dir=str(config_dir)):
            cfg = compose(config_name='config', overrides=overrides)
            dm = hydra.utils.instantiate(cfg.dataset, task='panoptic')
        dm.prepare_data()
        dm.setup('fit')
        return dm


    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=base_dataset')
        return overrides