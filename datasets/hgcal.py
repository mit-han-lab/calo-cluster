import os
import os.path as osp

import numpy as np
from numba import jit

from modules.efficient_minkowski import sparse_collate, sparse_quantize

__all__ = ['HGCAL']


class HGCALDataset:
    def __init__(self, root, split, voxel_size):
        self.voxel_size = voxel_size

        self.events = []
        for event in sorted(os.listdir(root)):
            if event.startswith('event'):
                self.events.append(osp.join(root, event))

        # TODO: check whether there is correlation between events
        pivot = int(len(self.events) * 0.9)
        assert split in ['train', 'test']
        if split == 'train':
            self.events = self.events[:pivot]
        else:
            self.events = self.events[pivot:]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = np.load(self.events[index])
        block, labels_ = event['x'], event['y']

        pc_ = np.round(block[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        return pc, feat, labels, labels_, inverse_map

    @staticmethod
    def collate_fn(tbl):
        locs, feats, labels, block_labels, invs = zip(*tbl)
        return sparse_collate(locs, feats, labels), block_labels, invs


class HGCAL(dict):
    def __init__(self, root, voxel_size, num_points):
        super(HGCAL, self).__init__({
            'train':
            HGCALDataset(root, 'train', voxel_size),
            'test':
            HGCALDataset(root, 'test', voxel_size)
        })
