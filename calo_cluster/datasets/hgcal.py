from dataclasses import dataclass

import numpy as np

from .base import BaseDataModule, BaseDataset


class HGCalDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        super().__init__(voxel_size, events, task)

    def _get_pc_feat_labels(self, index):
        event = np.load(self.events[index])
        if self.task == 'panoptic':
            block, labels_ = event['x'], event['y']
        elif self.task == 'semantic':
            block, labels_ = event['x'], event['y'][:, 0]
        elif self.task == 'instance':
            block, labels_ = event['x'], event['y'][:, 1]
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        pc_ = np.round(block[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block
        return pc_, feat_, labels_


@dataclass
class HGCalDataModule(BaseDataModule):

    def make_dataset(self, events) -> BaseDataset:
        return HGCalDataset(self.voxel_size, events, self.task)
