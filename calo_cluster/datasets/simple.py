from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.simple import SimpleDataModuleMixin
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)
from calo_cluster.datasets.pandas_data import PandasDataset


@dataclass
class SimpleDataset(SparseDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class SimpleOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class SimpleDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, SimpleDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SimpleDataset:
        kwargs = self.make_dataset_kwargs()
        return SimpleDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=simple')
        return overrides

@dataclass
class SimpleOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, SimpleDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SimpleOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return SimpleOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=simple_offset')
        return overrides