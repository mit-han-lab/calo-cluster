import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .calo import CaloDataModule, CaloDataset


class HCalTTPU200PFDataset(CaloDataset):
    def __init__(self, **kwargs):
        super().__init__(instance_label='PFcluster0Id', semantic_label='pf_hit', scale=False, mean=None, std=None, weight='energy', **kwargs)



@dataclass
class HCalTTPU200PFDataModule(CaloDataModule):
    noise_id: int
    min_hits_per_cluster: int
    min_cluster_energy: float

    def make_dataset(self, files: List[Path], split: str) -> HCalTTPU200PFDataset:
        return HCalTTPU200PFDataset(voxel_size=self.voxel_size, files=files, task=self.task, feats=self.feats, coords=self.coords, sparse=self.sparse)

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-1):
        ni = 0
        for f in tqdm(sorted(root_data_path.glob('*_pu.root'))):
            root_dir = uproot.open(f)
            root_events = root_dir.get('Events;1')
            df = pd.DataFrame()
            for k, v in root_events[b'RecHit'].items():
                df[k.decode('ascii').split('.')[1]] = v.array()

            for n in range(df.shape[0]):
                jagged_event = df.loc[n]
                df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
                flat_event = pd.DataFrame(df_dict)
                pf_noise_mask = (flat_event['PFcluster0Id'] == noise_id)
                flat_event['pf_hit'] = 0
                flat_event.loc[~pf_noise_mask, 'pf_hit'] = 1
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n

    @classmethod
    def _apply_transform(cls, event_path, min_cluster_energy, min_hits_per_cluster, noise_id, transformed_data_dir):
        event = pd.read_pickle(event_path)
        tracks = cls.get_clusters(event)

        tracks = tracks[(tracks['energy'] >= min_cluster_energy) | (
            tracks['clusterId'] == noise_id)]
        tracks = tracks[(tracks['nconstituents'] >= min_hits_per_cluster) | (
            tracks['clusterId'] == noise_id)]
        event = event[event['PFcluster0Id'].isin(tracks['clusterId'])]
        event = event.reset_index(drop=True)

        out_path = transformed_data_dir / event_path.name
        if event.shape[0] > 0:
            event.to_pickle(out_path)

    @classmethod
    def get_clusters(cls, event, cluster_col='PFcluster0Id'):
        """Find [eta, phi] (weighted by constituent energy) of clusters in event."""
        event['weta'] = event['eta'] * event['energy']
        event['wphi'] = event['phi'] * event['energy']
        grouped_event = event.groupby([cluster_col])
        angle_agg = grouped_event[
            ['weta', 'wphi']].agg(['sum'])
        energy_agg = grouped_event[
            ['energy']].agg(['sum', 'count'])
        energy = energy_agg[('energy', 'sum')]
        energy.name = 'energy'
        nconstituents = energy_agg[('energy', 'count')]
        nconstituents.name = 'nconstituents'
        eta = angle_agg[('weta', 'sum')] / energy
        eta.name = 'eta'
        phi = angle_agg[('wphi', 'sum')] / energy
        phi.name = 'phi'
        return pd.concat([energy, eta, phi, nconstituents], axis=1).reset_index().rename(columns={cluster_col: 'clusterId'})


    def get_transform_function(self):
            transformed_data_dir = self.raw_data_dir / f'min_energy_{self.min_cluster_energy}_min_hits_{self.min_hits_per_cluster}'
            return partial(self._apply_transform, min_cluster_energy=self.min_cluster_energy, min_hits_per_cluster=self.min_hits_per_cluster, noise_id=self.noise_id, transformed_data_dir=transformed_data_dir)