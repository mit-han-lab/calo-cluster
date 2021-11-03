from dataclasses import dataclass
from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.offset import (OffsetDataModuleMixin,
                                                 OffsetDatasetMixin)
from calo_cluster.datasets.mixins.scaled import (ScaledDataModuleMixin,
                                                 ScaledDatasetMixin)
from calo_cluster.datasets.mixins.sparse import (SparseDataModuleMixin,
                                                 SparseDatasetMixin)
from calo_cluster.datasets.mixins.vertex import (VertexDataModuleMixin,
                                                 VertexDatasetMixin)


@dataclass
class VertexDataset(SparseDatasetMixin, ScaledDatasetMixin, VertexDatasetMixin):
    pass


@dataclass
class VertexOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, VertexDatasetMixin):
    pass


@dataclass
class VertexDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, VertexDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> VertexDataset:
        kwargs = self.make_dataset_kwargs()
        return VertexDataset(files=files, **kwargs)


@dataclass
class VertexOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, VertexDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> VertexOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return VertexOffsetDataset(files=files, split=split, **kwargs)