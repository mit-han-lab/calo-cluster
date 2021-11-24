from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.kitti import (
    KITTIDataModuleMixin, KITTIDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)


@dataclass
class KITTIDataset(SparseDatasetMixin, ScaledDatasetMixin, KITTIDatasetMixin):
    pass

@dataclass
class KITTIOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, KITTIDatasetMixin):
    pass

@dataclass
class KITTIDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, KITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> KITTIDataset:
        kwargs = self.make_dataset_kwargs()
        return KITTIDataset(files=files, **kwargs)

@dataclass
class KITTIOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, KITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> KITTIOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return KITTIOffsetDataset(files=files, split=split, **kwargs)