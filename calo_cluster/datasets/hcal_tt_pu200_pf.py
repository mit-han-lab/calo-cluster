from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.hcal_tt_pu200 import \
    HCalTTPU200PFDataModuleMixin
from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)
from calo_cluster.datasets.pandas_data import PandasDataset
from tqdm.auto import tqdm


@dataclass
class HCalTTPU200PFDataset(SparseDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class HCalTTPU200PFOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class HCalTTPU200PFDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, HCalTTPU200PFDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalTTPU200PFDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalTTPU200PFDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_tt_pu200_pf')
        return overrides

@dataclass
class HCalTTPU200PFOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, HCalTTPU200PFDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalTTPU200PFOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalTTPU200PFOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_tt_pu200_pf_offset')
        return overrides