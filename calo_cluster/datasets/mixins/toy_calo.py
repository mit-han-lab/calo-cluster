from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Union

import numpy as np
import scipy
from calo_cluster.datasets.base import BaseDataModule
from calo_cluster.datasets.pandas_data import PandasDataset, PandasDataModuleMixin
from tqdm.auto import tqdm
import pandas as pd
from sklearn.datasets import make_blobs
from pathlib import Path

@dataclass
class ToyCaloDataModuleMixin(PandasDataModuleMixin, BaseDataModule):
    """Toy calo dataset designed to emulate hcal Zll_jets barrel (eta <= 1.5). 

    Parameters:
    nc_mu: mu of the poisson distribution for the number of clusters.
    np_mu: mu of the poisson distribution for the number of points per cluster.
    weight_np_mu: if true, weight np_mu by the cluster energy.
    p_s: sigma of the normal distribution for the points relative to the cluster center.
    center_dist: one of [uniform, hcal]. Distribution of the cluster centers.
    noise_dist: one of [uniform, hcal, None]. Distribution of the noise.
    """
    nc_mu: float
    np_mu: float
    weight_np_mu: bool
    p_s: float
    center_dist: str
    noise_dist: Union[str, None]
    def __post_init__(self):
        super().__post_init__()

        self.data_dir = Path(self.data_dir)
        if self.weight_np_mu:
            point_str = f'np{self.np_mu}w'
        else:
            point_str = f'np{self.np_mu}'
        self.data_dir = self.data_dir / f'nc_mu{self.nc_mu}_{point_str}_s{self.p_s}_noise={str(self.noise_dist)}_center={self.center_dist}'
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def prepare_data(self) -> None:
        if not self.data_exists():
            logging.info(
                f'dataset not found at {self.data_dir}.')
            self.make_data()

    @staticmethod
    def _generate_events(n_events, nc_mu: float, np_mu: float, p_s: float, center_dist: str, noise_dist: Union[str, None], weight_np_mu: bool):
        evts = []
        for i in tqdm(range(n_events)):
            n_clusters = 0
            while n_clusters == 0:
                n_clusters = scipy.stats.poisson.rvs(mu=nc_mu)
            cluster_energies = scipy.stats.expon.rvs(size=n_clusters)
            if center_dist == 'uniform':
                cluster_phi = scipy.stats.uniform.rvs(scale=2*math.pi, loc=-math.pi, size=n_clusters)
                cluster_eta = scipy.stats.uniform.rvs(scale=3, loc=-1.5, size=n_clusters)
            else:
                raise NotImplementedError()
            cluster_npoints = np.zeros(n_clusters, dtype=np.int32)
            for j in range(n_clusters):
                npoints = 0
                while npoints < 1:
                    mu = np_mu
                    if weight_np_mu:
                        mu = mu*cluster_energies[j]
                    npoints = scipy.stats.poisson.rvs(mu=mu)
                cluster_npoints[j] = npoints
            point_energies = []
            point_eta = []
            point_phi = []
            point_instance_id = []
            point_semantic_id = []
            for i, npoints in enumerate(cluster_npoints):
                p_phi = scipy.stats.norm.rvs(scale=p_s, size=npoints)
                p_eta = scipy.stats.norm.rvs(scale=p_s, size=npoints) 
                dist = (p_phi**2 + p_eta**2)**0.5
                p_phi += cluster_phi[i]
                p_eta += cluster_eta[i]
                if npoints > 1:
                    p_E = dist**(-1) / np.sum(dist**(-1)) * cluster_energies[i]
                else:
                    p_E = np.array(cluster_energies[i])
                point_energies.append(p_E)
                point_eta.append(p_eta)
                point_phi.append(p_phi)
                instance_id = np.full(npoints, i)
                point_instance_id.append(instance_id)
                point_semantic_id.append(np.full(npoints, 1))
            
            d = {'phi': point_phi, 'eta': point_eta, 'energy': point_energies, 'instance_id': point_instance_id, 'semantic_id': point_semantic_id}
            d = {k: np.concatenate(v) for k,v in d.items()}
            signal_event = pd.DataFrame(d)
            if noise_dist is not None:
                noise_n = scipy.stats.poisson.rvs(mu=18)
                noise_energy = scipy.stats.expon.rvs(scale=0.04, size=noise_n)
                if noise_dist == 'uniform':
                    noise_phi = scipy.stats.uniform.rvs(scale=2*math.pi, loc=-math.pi, size=noise_n)
                    noise_eta = scipy.stats.uniform.rvs(scale=3, loc=-1.5, size=noise_n)
                else:
                    raise NotImplementedError()
                noise_instance_id = np.full(noise_n, -1)
                noise_semantic_id = np.full(noise_n, 0)
                d = {'phi': noise_phi, 'eta': noise_eta, 'energy': noise_energy, 'instance_id': noise_instance_id, 'semantic_id': noise_semantic_id}
                noise_event = pd.DataFrame(d)
                evts.append(pd.concat((signal_event, noise_event)))
            else:
                evts.append(signal_event)
        return evts

    @staticmethod
    def generate(data_dir, nc_mu: float, np_mu: float, p_s: float, center_dist: str, noise_dist: Union[str, None], weight_np_mu: bool, n_events=10000) -> None:
        logging.info(f'Generating data at {data_dir}.')
        events = ToyCaloDataModuleMixin._generate_events(n_events, nc_mu, np_mu, p_s, center_dist, noise_dist, weight_np_mu)
        for i in tqdm(range(n_events)):
            df = events[i]
            event_path = data_dir / f'{i:05}.pkl'
            df.to_pickle(event_path)

    def make_data(self):
        self.generate(self.data_dir, nc_mu=self.nc_mu, np_mu=self.np_mu, p_s=self.p_s, center_dist=self.center_dist, noise_dist=self.noise_dist, weight_np_mu=self.weight_np_mu)

    def data_exists(self) -> bool:
        return len(set(self.data_dir.glob('*'))) != 0

if __name__ == '__main__':
    from pathlib import Path
    data_dir = Path('/data/toy_calo')
    ToyCaloDataModuleMixin.generate(data_dir, n_events=10000)
