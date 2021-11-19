
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import numpy as np

import pandas as pd

from calo_cluster.datasets.pandas_data import PandasDataModuleMixin
from calo_cluster.datasets.transformed_data import TransformedDataModule
import multiprocessing as mp

from tqdm.auto import tqdm

@dataclass
class CaloDataModule(PandasDataModuleMixin, TransformedDataModule):
    noise_id: int
    min_hits_per_cluster: int
    min_cluster_energy: float
    min_eta: float
    max_eta: float

    def __post_init__(self):
        super().__post_init__()
        if type(self.min_eta) is str:
            self.min_eta = None
        if type(self.max_eta) is str:
            self.max_eta = None
        if type(self.min_cluster_energy) is str:
            self.min_cluster_energy = None
        if type(self.min_hits_per_cluster) is str:
            self.min_hits_per_cluster = None
        if self.min_eta is None and self.max_eta is None and self.min_cluster_energy is None and self.min_hits_per_cluster is None:
            dir_name = 'raw'
        else:
            names = []
            if self.min_cluster_energy is not None:
                names.append(f'min_energy_{self.min_cluster_energy}')
            if self.min_hits_per_cluster is not None:
                names.append(f'min_hits_{self.min_hits_per_cluster}')
            if self.min_eta is not None:
                names.append(f'min_eta_{self.min_eta}')
            if self.max_eta is not None:
                names.append(f'max_eta_{self.max_eta}')
            dir_name = '_'.join(names)
        self.transformed_data_dir = self.data_dir / dir_name
        self.transformed_data_dir.mkdir(exist_ok=True, parents=True)

    @classmethod
    def _apply_transform(cls, event_path: Path, min_cluster_energy: float, min_hits_per_cluster: int, noise_id: int, min_eta: float, max_eta: float, transformed_data_dir: Path):
        event = pd.read_pickle(event_path)
        tracks = cls.get_clusters(event)

        tracks = tracks[(tracks['energy'] >= min_cluster_energy) | (
            tracks['clusterId'] == noise_id)]
        tracks = tracks[(tracks['nconstituents'] >= min_hits_per_cluster) | (
            tracks['clusterId'] == noise_id)]
        event = event[event['PFcluster0Id'].isin(tracks['clusterId'])]
        event = event.reset_index(drop=True)

        if min_eta is not None:
            event = event[event['eta'] > min_eta]
        if max_eta is not None:
            event = event[event['eta'] < max_eta]

        out_path = transformed_data_dir / event_path.name
        if event.shape[0] > 0:
            event.to_pickle(out_path)

    @classmethod
    def get_clusters(cls, event: pd.DataFrame, cluster_col='trackId', coords=['eta', 'phi'], weight_name='energy'):
        """Find [eta, phi] (weighted by constituent energy) of clusters in event."""
        wc_names = [f'w{c}' for c in coords]
        for wc, c in zip(wc_names, coords):
            event[wc] = event[c] * event[weight_name]
        grouped_event = event.groupby([cluster_col])
        coord_agg = grouped_event[wc_names].agg(['sum'])
        weight_agg = grouped_event[[weight_name]].agg(['sum', 'count'])
        weight_sum = weight_agg[(weight_name, 'sum')]
        weight_sum.name = weight_name
        nconstituents = weight_agg[(weight_name, 'count')]
        nconstituents.name = 'nconstituents'
        wcs = []
        for wc in wc_names:
            wcs.append(coord_agg[(wc, 'sum')] / weight_sum)
        return pd.concat([weight_sum, nconstituents] + wcs, axis=1).reset_index().rename(columns={cluster_col: 'clusterId'})

    def get_transform_function(self):
        return partial(self._apply_transform, min_cluster_energy=self.min_cluster_energy, min_hits_per_cluster=self.min_hits_per_cluster, min_eta=self.min_eta, max_eta=self.max_eta, noise_id=self.noise_id, transformed_data_dir=self.transformed_data_dir)


    @classmethod
    def merge_clusters(cls, event: pd.DataFrame, cluster_col='trackId', weight_name='energy', coords=['eta', 'phi'], threshold=0.1, noise_id=-99):
        noise = event[event[cluster_col] == noise_id].reset_index(drop=True)
        true_hits = event[event[cluster_col] != noise_id].reset_index(drop=True)
        
        while True:
            # Find delta R between each cluster, identify pairs that are mergeable and sort according to energy.
            tracks = cls.get_clusters(true_hits, cluster_col=cluster_col, weight_name=weight_name, coords=coords)
            eta = tracks[0].values
            phi = tracks[1].values
            dR2 = (np.expand_dims(eta, axis=1) - eta)**2 + (np.expand_dims(phi, axis=1) - phi)**2
            np.fill_diagonal(dR2, 1.0)
            mergeable = dR2 < threshold**2
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
                new_hits.loc[true_hits[cluster_col]==k, cluster_col] = v
            true_hits = new_hits

        # Fix ids.
        track_ids = true_hits[cluster_col].unique()
        for i, track_id in enumerate(track_ids):
            true_hits.loc[true_hits[cluster_col]==track_id, cluster_col] = i
        merged_event = pd.concat([noise, true_hits]).sample(frac=1).reset_index(drop=True)
        return merged_event
    
    @classmethod
    def _merge_clusters(cls, event_path, data_dir, cluster_col='trackId', weight_name='energy', coords=['eta', 'phi'], threshold=0.1, noise_id=-99):
        event = pd.read_pickle(event_path)
        merged_event = cls.merge_clusters(event, cluster_col, weight_name, coords, threshold, noise_id)
        merged_event_path = data_dir / event_path.name
        merged_event.to_pickle(merged_event_path)
    
    @classmethod
    def merge_events(cls, raw_data_dir, data_dir, cluster_col='trackId', weight_name='energy', coords=['eta', 'phi'], threshold=0.1, noise_id=-99, ncpus=10):
        with mp.Pool(ncpus) as p:
            raw_event_paths = [f for f in raw_data_dir.glob('*.pkl')]
            with tqdm(total=len(raw_event_paths)) as pbar:
                for _ in p.imap_unordered(partial(cls._merge_clusters, data_dir=data_dir, cluster_col=cluster_col, weight_name=weight_name, coords=coords, threshold=threshold, noise_id=noise_id), raw_event_paths):
                    pbar.update()