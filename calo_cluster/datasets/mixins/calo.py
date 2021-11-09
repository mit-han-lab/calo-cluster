
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import uproot
from calo_cluster.datasets.pandas_data import PandasDataModuleMixin, PandasDataset
from calo_cluster.datasets.transformed_data import TransformedDataModule
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
        if self.min_eta is None and self.max_eta is None:
            self.transformed_data_dir = transformed_data_dir = self.data_dir / \
                f'min_energy_{self.min_cluster_energy}_min_hits_{self.min_hits_per_cluster}'
        else:
            self.transformed_data_dir = transformed_data_dir = self.data_dir / \
                f'min_energy_{self.min_cluster_energy}_min_hits_{self.min_hits_per_cluster}_eta_{self.min_eta}_{self.max_eta}'
        transformed_data_dir.mkdir(exist_ok=True, parents=True)

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
    def get_clusters(cls, event: pd.DataFrame, cluster_col='PFcluster0Id', coords=['eta', 'phi'], weight_name='energy'):
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