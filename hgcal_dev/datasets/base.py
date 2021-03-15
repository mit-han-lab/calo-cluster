import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchsparse.utils import sparse_quantize
from hgcal_dev.utils.sparse_collate import sparse_collate_fn
from torchsparse import SparseTensor

from dataclasses import dataclass

@dataclass
class BaseDataset(Dataset):
    "Base torch dataset."
    voxel_size: float
    events: list
    task: str
    feats: list = None
    coords: list = None
    class_label: str = 'class'
    instance_label: str = 'instance'
    ignore_label: int = -1
    scale: bool = False
    std: list = None
    mean: list = None

    def __len__(self):
        return len(self.events)

    def _get_pc_feat_labels(self, index):
        event = pd.read_pickle(self.events[index])
        feat_ = event[self.feats].to_numpy()
        if self.task == 'panoptic':
            labels_ = event[[self.class_label, self.instance_label]].to_numpy()
        elif self.task == 'semantic':
            labels_ = event[self.class_label].to_numpy()
        elif self.task == 'instance':
            labels_ = event[self.instance_label].to_numpy(
            )
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        pc_ = np.round(event[self.coords].to_numpy() / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        if self.scale:
            feat_ = (feat_ - np.array(self.mean)) / np.array(self.std)
        return pc_, feat_, labels_

    def __getitem__(self, index):
        pc_, feat_, labels_ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    ignore_label=self.ignore_label)
        pc = pc_[inds]
        feat = feat_[inds]
        features = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        return {'features': features, 'labels': labels, 'labels_mapped': labels_, 'inverse_map': inverse_map}

    def get_inds_labels(self, index):
        pc_, feat_, labels_ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    ignore_label=self.ignore_label)
        return inds, labels

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)