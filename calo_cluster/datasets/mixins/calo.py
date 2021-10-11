
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import uproot
from calo_cluster.datasets.pandas_data import PandasDataModule
from tqdm import tqdm


@dataclass
class CaloDataModule(PandasDataModule):
    noise_id: int
    min_hits_per_cluster: int
    min_cluster_energy: float

    @classmethod
    def _apply_transform(cls, event_path: Path, min_cluster_energy: float, min_hits_per_cluster: int, noise_id: int, transformed_data_dir: Path):
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
    def get_clusters(cls, event: pd.DataFrame, cluster_col='PFcluster0Id'):
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
        transformed_data_dir = self.raw_data_dir / \
            f'min_energy_{self.min_cluster_energy}_min_hits_{self.min_hits_per_cluster}'
        return partial(self._apply_transform, min_cluster_energy=self.min_cluster_energy, min_hits_per_cluster=self.min_hits_per_cluster, noise_id=self.noise_id, transformed_data_dir=transformed_data_dir)
