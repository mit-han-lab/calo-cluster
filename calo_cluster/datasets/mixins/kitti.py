import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from calo_cluster.datasets.base import BaseDataModule, BaseDataset

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


@dataclass
class KITTIDatasetMixin(BaseDataset):
    split: str
    label_map: np.array

    def __post_init__(self):
        self.angle = 0.0
        super().__post_init__()

    def set_angle(self, angle: float):
        self.angle = angle

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]

        label_file = str(self.files[index]).replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block.shape[0]).astype(np.int32)

        semantic_labels = self.label_map[all_labels & 0xFFFF].astype(
                np.int32)
        instance_labels = ((all_labels >> 16) & 0xFFFF).astype(np.int32)

        return_dict = {'features': block, 'coordinates': block[:, :3], 'semantic_labels': semantic_labels, 'instance_labels': instance_labels}
        return return_dict


@dataclass
class KITTIDataModuleMixin(BaseDataModule):
    data_dir: str

    def __post_init__(self):
        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        assert self.num_classes == cnt
        super().__post_init__()

    def _get_files(self, seqs):
        files = []
        for seq in seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.data_dir, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.data_dir, seq, 'velodyne', x) for x in seq_files
            ]
            files.extend(seq_files)

        return [Path(f) for f in files]

    def train_val_test_split(self):
        train_seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        val_seqs = ['08']
        test_seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        train_files = self._get_files(train_seqs)
        val_files = self._get_files(val_seqs)
        test_files = self._get_files(test_seqs)

        train_files = train_files[:int(len(train_files)*self.event_frac)]
        val_files = val_files[:int(len(val_files)*self.event_frac)]
        test_files = test_files[:int(len(test_files)*self.event_frac)]
           
        return train_files, val_files, test_files

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'label_map': self.label_map
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs