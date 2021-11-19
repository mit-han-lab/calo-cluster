from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.hgcal_taus import (
    HGCalTausDataModuleMixin, HGCalTausMixin)
from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)

# see https://github.com/cms-pepr/pytorch_cmspepr/blob/68ef83386c6656a388815031f4679c19a3ca62db/torch_cmspepr/dataset.py#L104

@dataclass
class HGCalTausDataset(SparseDatasetMixin, ScaledDatasetMixin, HGCalTausMixin):
    pass

@dataclass
class HGCalTausOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, HGCalTausMixin):
    pass

@dataclass
class HGCalTausDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, HGCalTausDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HGCalTausDataset:
        kwargs = self.make_dataset_kwargs()
        return HGCalTausDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hgcal_taus')
        return overrides

@dataclass
class HGCalTausOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, HGCalTausDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HGCalTausOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return HGCalTausOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hgcal_taus_offset')
        return overrides