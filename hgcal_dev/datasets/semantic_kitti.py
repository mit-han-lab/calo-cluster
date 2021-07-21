import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import uproot
from torch.utils.data import DataLoader
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from tqdm import tqdm

from .base import BaseDataModule, BaseDataset

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
class SemanticKITTIDataset:
    root: str
    voxel_size: float
    split: str
    task: str
    num_points: int
    sparse: bool

    def __post_init__(self):
        if not self.sparse:
            self.collate_fn = None
        self.weight = None
        self.seqs = []
        if self.split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

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
        self.num_classes = cnt
        self.angle = 0.0

    @property
    def events(self) -> list:
        return [Path(f) for f in self.files]

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
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
        if self.sparse:
            pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        else:
            pc_ = block[:, :3].astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        if self.task == 'semantic':
            labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
        elif self.task == 'instance':
            labels_ = ((all_labels >> 4) & 0xFFFF).astype(np.int64)
        elif self.task == 'panoptic':
            semantic_labels = self.label_map[all_labels & 0xFFFF].astype(
                np.int64)
            instance_labels = ((all_labels >> 4) & 0xFFFF).astype(np.int64)
            labels_ = np.stack((semantic_labels, instance_labels), axis=-1)
        feat_ = block

        if self.sparse:
            _, inds, inverse_map = sparse_quantize(pc_,
                                                return_index=True,
                                                return_inverse=True)

            if 'train' in self.split:
                if len(inds) > self.num_points:
                    inds = np.random.choice(inds, self.num_points, replace=False)

            pc = pc_[inds]
            feat = feat_[inds]
            labels = labels_[inds]
            lidar = SparseTensor(feat, pc)
            labels = SparseTensor(labels, pc)
            inverse_map = SparseTensor(inverse_map, pc_)

            return {
                'features': lidar,
                'labels': labels,
                'inverse_map': inverse_map,
                'file_name': self.files[index]
            }
        else:
            if len(feat_) > self.num_points:
                inds = np.random.choice(np.arange(len(feat_)), self.num_points, replace=False)
                feat_ = feat_[inds]
                labels_ = labels_[inds]
            return {
                    'features': feat_,
                    'labels': labels_
            }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)


@dataclass
class SemanticKITTIDataModule(pl.LightningDataModule):
    root: str
    voxel_size: float
    num_points: int
    seed: int
    batch_size: int
    num_epochs: int
    num_workers: int
    num_classes: int
    task: str
    ignore_label: int
    num_features: int
    sparse: bool

    def __post_init__(self):
        super().__init__()

    def setup(self, stage=None) -> None:

        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = SemanticKITTIDataset(
                self.root, self.voxel_size, num_points=self.num_points, split='train', task=self.task, sparse=self.sparse)
            self.val_dataset = SemanticKITTIDataset(
                self.root, self.voxel_size, num_points=self.num_points, split='val', task=self.task, sparse=self.sparse)
        if stage == 'test' or stage is None:
            self.test_dataset = SemanticKITTIDataset(
                self.root, self.voxel_size, num_points=self.num_points, split='test', task=self.task, sparse=self.sparse)

    def dataloader(self, dataset: BaseDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    def voxel_occupancy(self):
        self.batch_size = 1
        dataloader = self.train_dataloader()
        voxel_occupancies = np.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            voxel_occupancies[i] = len(batch['inverse_map'].C) / len(batch['features'].C) 

        return voxel_occupancies
