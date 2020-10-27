import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

from ..modules.efficient_minkowski import sparse_collate, sparse_quantize


class BaseDataset(Dataset):
    def __init__(self, voxel_size, events, task, feats=None, coords=None, class_label='class', instance_label='instance'):
        self.voxel_size = voxel_size
        self.events = events
        self.task = task
        self.class_label = class_label
        self.instance_label = instance_label
        self.feats = feats
        self.coords = coords

    def __len__(self):
        return len(self.events)

    def _get_pc_feat_labels(self, index):
        event = pd.read_pickle(self.events[index])
        if self.task == 'panoptic':
            block, labels_ = event[self.feats], event[[
                self.class_label, self.instance_label]].to_numpy()
        elif self.task == 'semantic':
            block, labels_ = event[self.feats], event[self.class_label].to_numpy(
            )
        elif self.task == 'instance':
            block, labels_ = event[self.feats], event[self.instance_label].to_numpy(
            )
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        pc_ = np.round(block[self.coords].to_numpy() / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block.to_numpy()
        return pc_, feat_, labels_

    def __getitem__(self, index):
        pc_, feat_, labels_ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        return pc, feat, labels, labels_, inverse_map

    def get_inds_labels(self, index):
        pc_, feat_, labels_ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)
        return inds, labels

    @staticmethod
    def collate_fn(tbl):
        locs, feats, labels, block_labels, invs = zip(*tbl)
        return sparse_collate(locs, feats, labels), block_labels, invs
