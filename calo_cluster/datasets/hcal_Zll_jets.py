from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.hcal_Zll_jets import (
    HCalZllJetsDataModuleMixin, HCalZllJetsMixin)
from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)


@dataclass
class HCalZllJetsDataset(SparseDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass

@dataclass
class HCalZllJetsOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass

@dataclass
class HCalZllJetsDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, HCalZllJetsDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalZllJetsDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalZllJetsDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_Zll_jets')
        return overrides

@dataclass
class HCalZllJetsOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, HCalZllJetsDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalZllJetsOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalZllJetsOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_Zll_jets_offset')
        return overrides