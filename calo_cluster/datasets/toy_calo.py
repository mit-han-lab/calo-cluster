from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)
from calo_cluster.datasets.mixins.toy_calo import ToyCaloDataModuleMixin
from calo_cluster.datasets.pandas_data import PandasDataset


@dataclass
class ToyCaloDataset(SparseDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class ToyCaloOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class ToyCaloDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, ToyCaloDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> ToyCaloDataset:
        kwargs = self.make_dataset_kwargs()
        return ToyCaloDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=toy_calo')
        return overrides

@dataclass
class ToyCaloOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, ToyCaloDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> ToyCaloOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return ToyCaloOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=toy_calo_offset')
        return overrides