from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.semantic_kitti import (
    SemanticKITTIDataModuleMixin, SemanticKITTIDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)


@dataclass
class SemanticKITTIDataset(SparseDatasetMixin, ScaledDatasetMixin, SemanticKITTIDatasetMixin):
    pass

@dataclass
class SemanticKITTIOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, SemanticKITTIDatasetMixin):
    pass

@dataclass
class SemanticKITTIDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, SemanticKITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SemanticKITTIDataset:
        kwargs = self.make_dataset_kwargs()
        return SemanticKITTIDataset(files=files, **kwargs)

@dataclass
class SemanticKITTIOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, SemanticKITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SemanticKITTIOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return SemanticKITTIOffsetDataset(files=files, split=split, **kwargs)