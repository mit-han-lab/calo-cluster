from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.hcal_tt import (
    HCalTTDataModuleMixin, HCalTTMixin)
from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)


@dataclass
class HCalTTDataset(SparseDatasetMixin, ScaledDatasetMixin, HCalTTMixin):
    pass

@dataclass
class HCalTTOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, HCalTTMixin):
    pass

@dataclass
class HCalTTDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, HCalTTDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalTTDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalTTDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_tt')
        return overrides

@dataclass
class HCalTTOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, HCalTTDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalTTOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalTTOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_tt_offset')
        return overrides