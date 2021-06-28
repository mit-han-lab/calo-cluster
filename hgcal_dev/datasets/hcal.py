import multiprocessing as mp
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .base import BaseDataModule, BaseDataset


class HCalDataset(BaseDataset):
    def __init__(self, voxel_size, events, task, instance_label, feats, coords):
        if instance_label == 'truth':
            instance_label = 'trackId'
        elif instance_label == 'antikt':
            instance_label = 'RHAntiKtCluster'
        else:
            raise RuntimeError()
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label=instance_label, weight='energy')

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
        if self.weight is not None:
            weights_ = event[self.weight].to_numpy()
        else:
            weights_ = None
        return pc_, feat_, labels_, weights_


@dataclass
class HCalDataModule(BaseDataModule):
    min_cluster_energy: float
    min_hits_per_cluster: int
    instance_label: str
    noise_id: int
    feats: list
    coords: list

    def __post_init__(self):
        super().__post_init__()
        self.transformed_data_dir = self.data_dir / \
            f'min_energy_{self.min_cluster_energy}_min_hits_{self.min_hits_per_cluster}'
        self.transformed_data_dir.mkdir(parents=True, exist_ok=True)

    def make_dataset(self, events) -> BaseDataset:
        return HCalDataset(self.voxel_size, events, self.task, self.instance_label, self.feats, self.coords)

    @classmethod
    def _apply_transform(cls, event_path, min_cluster_energy, min_hits_per_cluster, noise_id, transformed_data_dir):
        event = pd.read_pickle(event_path)
        tracks = cls.get_clusters(event)

        tracks = tracks[(tracks['energy'] >= min_cluster_energy) | (
            tracks['clusterId'] == noise_id)]
        tracks = tracks[(tracks['nconstituents'] >= min_hits_per_cluster) | (
            tracks['clusterId'] == noise_id)]
        event = event[event['trackId'].isin(tracks['clusterId'])]
        event = event.reset_index(drop=True)

        out_path = transformed_data_dir / event_path.name
        if event.shape[0] > 0:
            event.to_pickle(out_path)

    def get_transform_function(self):
        return partial(self._apply_transform, min_cluster_energy=self.min_cluster_energy, min_hits_per_cluster=self.min_hits_per_cluster, noise_id=self.noise_id, transformed_data_dir=self.transformed_data_dir)

    @classmethod
    def get_clusters(cls, event, cluster_col='trackId'):
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

    @classmethod
    def merge_event(cls, event_path, data_dir, granularity=0.015, noise_id=-99):
        event = pd.read_pickle(event_path)
        noise = event[event['trackId'] == noise_id].reset_index(drop=True)
        true_hits = event[event['trackId'] != noise_id].reset_index(drop=True)

        while True:
            # Find delta R between each cluster, identify pairs that are mergeable and sort according to energy.
            tracks = cls.get_clusters(true_hits)
            eta = tracks['eta'].values
            phi = tracks['phi'].values
            dR2 = (np.expand_dims(eta, axis=1) - eta)**2 + \
                (np.expand_dims(phi, axis=1) - phi)**2
            np.fill_diagonal(dR2, 1.0)
            mergeable = dR2 < granularity**2
            sorted_indices = np.argsort(tracks['energy'].values)[::-1]

            # Keep the highest energy pairs.
            X, Y = np.where(mergeable[sorted_indices])
            new_ids = {}
            track_ids = tracks['clusterId'].values
            for x, y in zip(X, Y):
                x = track_ids[sorted_indices[x]]
                y = track_ids[y]
                if y not in new_ids and x not in new_ids:
                    new_ids[y] = x
            if len(new_ids) == 0:
                break

            # Assign the new ids
            new_hits = true_hits.copy()
            for k, v in new_ids.items():
                new_hits.loc[true_hits['trackId'] == k, 'trackId'] = v
            true_hits = new_hits

        # Fix ids.
        track_ids = true_hits['trackId'].unique()
        for i, track_id in enumerate(track_ids):
            true_hits.loc[true_hits['trackId'] == track_id, 'trackId'] = i
        merged_event = pd.concat([noise, true_hits]).sample(
            frac=1).reset_index(drop=True)
        merged_event_path = data_dir / event_path.name
        merged_event.to_pickle(merged_event_path)

    @classmethod
    def merge_events(cls, raw_data_dir, data_dir, granularity=0.05, ncpus=32):
        with mp.Pool(ncpus) as p:
            raw_event_paths = [f for f in raw_data_dir.glob('*.pkl')]
            with tqdm(total=len(raw_event_paths)) as pbar:
                for _ in p.imap_unordered(partial(cls.merge_event, data_dir=data_dir, granularity=granularity), raw_event_paths):
                    pbar.update()

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir, noise_id=-99):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.rootio.open(f)
            root_events = root_dir.get('Events;1')
            df = pd.DataFrame()
            for k, v in root_events[b'RecHit'].items():
                df[k.decode('ascii').split('.')[1]] = v.array()

            for n in tqdm(range(df.shape[0])):
                jagged_event = df.loc[n]
                df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
                flat_event = pd.DataFrame(df_dict)
                flat_event.astype({'hit': int})
                hit_mask = (flat_event['genE'] > 0.2)
                noise_mask = ~hit_mask
                flat_event['hit'] = hit_mask.astype(int)
                flat_event.loc[noise_mask, 'trackId'] = noise_id
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni = n + 1

