from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .calo import CaloDataset
from .hcal_tt_pu200_pf import HCalTTPU200PFDataModule


class HCalZllJetsDataset(CaloDataset):
    def __init__(self, instance_label, **kwargs):
        if instance_label == 'truth':
            raise NotImplementedError()
            #instance_label = 'trackId'
        elif instance_label == 'antikt':
            raise NotImplementedError()
            #instance_label = 'RHAntiKtCluster_reco'
        elif instance_label == 'pf':
            instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()
        scale = False
        mean = None
        std = None
        super().__init__(semantic_label='pf_hit', instance_label=instance_label, scale=scale, mean=mean, std=std, weight='energy', **kwargs)


@dataclass
class HCalZllJetsDataModule(HCalTTPU200PFDataModule):
    instance_label: str

    def make_dataset(self, files: List[Path], split: str) -> HCalZllJetsDataset:
        return HCalZllJetsDataset(files=files, voxel_size=self.voxel_size, task=self.task, feats=self.feats, coords=self.coords, instance_label=self.instance_label, sparse=self.sparse)

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-99, pf_noise_id=-1):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.rootio.open(f)
            root_events = root_dir.get('Events;1')
            df = pd.DataFrame()
            for k, v in tqdm(root_events[b'RecHit'].items()):
                df[k.decode('ascii').split('.')[1]] = v.array()

            for n in tqdm(range(df.shape[0])):
                jagged_event = df.loc[n]
                df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
                flat_event = pd.DataFrame(df_dict)
                pf_noise_mask = (flat_event['PFcluster0Id'] == noise_id)
                flat_event['pf_hit'] = 0
                flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
                flat_event.astype({'hit': int})
                hit_mask = (flat_event['genE'] > 0.2)
                noise_mask = ~hit_mask
                flat_event['hit'] = hit_mask.astype(int)
                flat_event.loc[noise_mask, 'trackId'] = noise_id
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni = n + 1
