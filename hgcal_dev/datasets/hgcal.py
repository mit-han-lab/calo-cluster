import logging
import os
import os.path as osp
import random
import tarfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..modules.efficient_minkowski import sparse_collate, sparse_quantize
from ..utils.comm import get_rank


class HGCalDataset(Dataset):
    def __init__(self, voxel_size, events, label_type):
        self.voxel_size = voxel_size
        self.events = events
        self.label_type = label_type

    def __len__(self):
        return len(self.events)

    def _get_pc_feat_labels(self, index):
        event = np.load(self.events[index])
        if self.label_type == 'class_and_instance':
            block, labels_ = event['x'], event['y']
        elif self.label_type == 'class':
            block, labels_ = event['x'], event['y'][:, 0]
        elif self.label_type == 'instance':
            block, labels_ = event['x'], event['y'][:, 1]
        else:
            raise RuntimeError(f'Unknown label_type = "{self.label_type}"')
        pc_ = np.round(block[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block
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


class HGCalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_train_events: int, num_test_events: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, data_url: str = 'https://cernbox.cern.ch/index.php/s/ocpNBUygDnMP3tx/download', download: bool = False, seed: int = None, num_events: int = -1, label_type: str = 'class', num_classes: int = 4, noise_level: float = 1.0, noise_seed: int = 31):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_train_events = num_train_events
        self.num_test_events = num_test_events
        self.num_events = num_events
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.voxel_size = voxel_size
        self.data_dir = Path(data_dir)
        self.data_url = data_url
        self._download = download
        self.seed = seed
        self.raw_data_dir = self.data_dir / '1.0_noise'
        self.compressed_data_path = self.data_dir / 'data.tar.gz'
        self.label_type = label_type
        assert 0.0 <= noise_level <= 1.0
        self.noise_seed = noise_seed
        self.noise_level = noise_level
        self.noisy_data_dir = self.data_dir / f'{noise_level}_noise'

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_data_dir.mkdir(parents=True, exist_ok=True)

        self._raw_events = None
        self._events = None

    @property
    def raw_events(self) -> list:
        if self._raw_events is None:
            self._raw_events = []
            self._raw_events.extend(sorted(self.raw_data_dir.glob('*.npz')))
        return self._raw_events

    @property
    def events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(sorted(self.noisy_data_dir.glob('*.npz')))
        return self._events

    def is_downloaded(self) -> bool:
        return len(set(self.data_dir.glob('data.tar.gz'))) != 0

    def is_extracted(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def noisy_data_exists(self) -> bool:
        return len(set(self.noisy_data_dir.glob('*'))) != 0

    def download(self) -> Path:
        logging.info(
            f'Downloading data to {self.data_dir} (this may take a few minutes).')
        with requests.get(self.data_url, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            compressed_data_path = self.data_dir / 'data.tar.gz'
            with compressed_data_path.open(mode='wb') as f:
                pbar = tqdm(total=int(r.headers['Content-Length']))
                for chunk in r.iter_content(chunk_size=1024**2):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logging.info('download complete.')

    def extract(self) -> None:
        logging.info(f'Extracting data to {self.raw_data_dir}.')
        tar = tarfile.open(self.compressed_data_path, "r:gz")
        for member in tqdm(tar.getmembers()):
            member.name = os.path.basename(member.name)
            tar.extract(member, self.raw_data_dir)
        tar.close()

    def make_noisy_data(self) -> None:
        if self.noise_level == 1.0:
            return
        logging.info(f'Saving noisy data to {self.noisy_data_dir}.')
        for event_path in tqdm(self.raw_events):
            event = np.load(event_path)
            x, y = event['x'], event['y']
            non_noise_indices = np.where(y[:, 0] != 0)[0]
            noise_indices = np.where(y[:, 0] == 0)[0]
            np.random.seed(self.noise_seed)
            selected_noise_indices = np.random.choice(
                noise_indices, size=int(noise_indices.shape[0]*self.noise_level))
            selected_indices = np.concatenate(
                (non_noise_indices, selected_noise_indices))
            if selected_indices.shape[0] > 0:
                x_selected = x[selected_indices]
                y_selected = y[selected_indices]
                noisy_event_path = self.noisy_data_dir / event_path.stem
                np.savez(noisy_event_path, x=x_selected, y=y_selected)

    def prepare_data(self) -> None:
        if not self.noisy_data_exists():
            logging.info(f'Noisy dataset not found at {self.noisy_data_dir}.')
            if not self.is_extracted():
                logging.info(f'Raw dataset not found at {self.raw_data_dir}.')
                if not self.is_downloaded():
                    logging.info(
                        f'Downloaded dataset not found at {self.compressed_data_path}.')
                    if self._download:
                        self.download()
                    else:
                        logging.error('download=false, aborting!')
                        raise RuntimeError()
                self.extract()
            self.make_noisy_data()

    def setup(self, stage) -> None:
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        events = self.events
        if self.num_events != -1:
            events = events[:self.num_events]
        if stage == 'fit' or stage is None:
            train_events = events[:self.num_train_events]
            logging.debug(f'num training events={len(train_events)}')
            val_events = events[self.num_train_events:-
                                self.num_test_events]
            logging.debug(f'num val events={len(val_events)}')
            self.train_dataset = HGCalDataset(
                self.voxel_size, train_events, self.label_type)
            self.val_dataset = HGCalDataset(
                self.voxel_size, val_events, self.label_type)
        if stage == 'test' or stage is None:
            test_events = events[-self.num_test_events:]
            logging.debug(f'num test events={len(test_events)}')
            self.test_dataset = HGCalDataset(
                self.voxel_size, test_events, self.label_type)

    def dataloader(self, dataset: HGCalDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=HGCalDataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)


if __name__ == '__main__':
    data_module = HGCalDataModule(
        1, 10000, 1000, 15, 8, 1.0, '/global/cscratch1/sd/schuya/hgcal-dev/data/hgcal', noise_level=0.0)
    data_module.prepare_data()
    data_module.setup('fit')
    dataloader = data_module.train_dataloader()
    print(dataloader.dataset.events[2168])
    breakpoint()
    dataloader.dataset[2168]
    for i, data in enumerate(tqdm(dataloader)):
        pass
