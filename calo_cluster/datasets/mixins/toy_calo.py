from dataclasses import dataclass
import logging
import math
from typing import Any, Dict

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
    """Toy calo dataset designed to emulate hcal Zll_jets. 

    Parameters:
    nc_mu: mu of the poisson distribution for the number of clusters.
    np_mu: mu of the poisson distribution for the number of points per cluster (weighted by cluster energy).
    p_s: sigma of the normal distribution for the points relative to the cluster center."""
    nc_mu: float
    np_mu: float
    p_s: float
    include_noise: bool

    def __post_init__(self):
        super().__post_init__()

        self.data_dir = Path(self.data_dir)
        self.data_dir = self.data_dir / f'nc_mu{self.nc_mu}_np{self.np_mu}_s{self.p_s}_noise={self.include_noise}'
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def prepare_data(self) -> None:
        if not self.data_exists():
            logging.info(
                f'dataset not found at {self.data_dir}.')
            self.make_data()

    @staticmethod
    def _generate_events(n_events, nc_mu: float = 10.0, np_mu: float = 1.0, p_s: float = 5.0, include_noise: bool = True):
        evts = []
        for i in tqdm(range(n_events)):
            n_clusters = 0
            while n_clusters == 0:
                n_clusters = scipy.stats.poisson.rvs(mu=nc_mu)
            cluster_energies = scipy.stats.expon.rvs(size=n_clusters)
            cluster_phi = scipy.stats.uniform.rvs(scale=2*math.pi, loc=-math.pi, size=n_clusters)
            cluster_eta = scipy.stats.arcsine.rvs(scale=3, loc=-1.5, size=n_clusters)
            cluster_r = []
            cluster_npoints = []
            for j in range(n_clusters):
                npoints = 0
                while npoints < 1:
                    npoints = scipy.stats.poisson.rvs(mu=np_mu*cluster_energies[j])
                cluster_npoints.append(npoints)
                r = 0
                while r < 1:
                    r = scipy.stats.poisson.rvs(mu=1)
                if r == 1:
                    r = 180
                elif r == 2:
                    r = 190
                elif r == 3:
                    r = 196
                elif r == 4:
                    r = 214
                r = r + scipy.stats.norm.rvs()
                cluster_r.append(r)
            cluster_r = np.array(cluster_r)
            cluster_npoints = np.array(cluster_npoints)
            cluster_theta = 2 * np.arctan(np.exp(-cluster_eta))
            cluster_x = cluster_r * np.cos(cluster_phi) * np.sin(cluster_theta)
            cluster_y = cluster_r * np.sin(cluster_phi) * np.sin(cluster_theta)
            cluster_z = cluster_r * np.cos(cluster_theta)
            point_energies = []
            point_x = []
            point_y = []
            point_z = []
            point_eta = []
            point_phi = []
            point_r = []
            point_instance_id = []
            point_semantic_id = []
            for i, npoints in enumerate(cluster_npoints):
                p_x = scipy.stats.norm.rvs(scale=p_s, size=npoints)
                p_y = scipy.stats.norm.rvs(scale=p_s, size=npoints)
                p_z = scipy.stats.norm.rvs(scale=p_s, size=npoints)
                dist = (p_x**2 + p_y**2 + p_z**2)**0.5
                p_E = dist / np.sum(dist) * cluster_energies[i]
                point_energies.append(p_E)
                p_x += cluster_x[i]
                p_y += cluster_y[i]
                p_z += cluster_z[i]
                point_x.append(p_x)
                point_y.append(p_y)
                point_z.append(p_z)
                p_phi = np.arctan2(p_y, p_x)
                p_r = (p_x**2 + p_y**2 + p_z**2)**0.5
                p_theta = np.arccos(p_z/p_r)
                p_eta = -np.log(np.tan(p_theta/2))
                point_eta.append(p_eta)
                point_phi.append(p_phi)
                point_r.append(p_r)
                instance_id = np.full(npoints, i)
                point_instance_id.append(instance_id)
                point_semantic_id.append(np.full(npoints, 1))
            
            d = {'r': point_r, 'phi': point_phi, 'eta': point_eta, 'z': point_z, 'y': point_y, 'x': point_x, 'energy': point_energies, 'instance_id': point_instance_id, 'semantic_id': point_semantic_id}
            d = {k: np.concatenate(v) for k,v in d.items()}
            signal_event = pd.DataFrame(d)
            if include_noise:
                noise_n = scipy.stats.poisson.rvs(mu=18)
                noise_r = []
                for j in range(noise_n):
                    r = 0
                    while r < 1:
                        r = scipy.stats.poisson.rvs(mu=1)
                    if r == 1:
                        r = 180
                    elif r == 2:
                        r = 190
                    elif r == 3:
                        r = 196
                    elif r == 4:
                        r = 214
                    r = r + scipy.stats.norm.rvs()
                    noise_r.append(r)
                noise_r = np.array(noise_r)
                noise_energy = scipy.stats.expon.rvs(scale=0.04, size=noise_n)
                noise_phi = scipy.stats.uniform.rvs(scale=2*math.pi, loc=-math.pi, size=noise_n)
                noise_eta = scipy.stats.arcsine.rvs(scale=3, loc=-1.5, size=noise_n)
                noise_theta = 2 * np.arctan(np.exp(-noise_eta))
                noise_x = noise_r * np.cos(noise_phi) * np.sin(noise_theta)
                noise_y = noise_r * np.sin(noise_phi) * np.sin(noise_theta)
                noise_z = noise_r * np.cos(noise_theta)
                noise_instance_id = np.full(noise_n, -1)
                noise_semantic_id = np.full(noise_n, 0)
                d = {'r': noise_r, 'phi': noise_phi, 'eta': noise_eta, 'z': noise_z, 'y': noise_y, 'x': noise_x, 'energy': noise_energy, 'instance_id': noise_instance_id, 'semantic_id': noise_semantic_id}
                noise_event = pd.DataFrame(d)
                evts.append(pd.concat((signal_event, noise_event)))
            else:
                evts.append(signal_event)
        return evts

    @staticmethod
    def generate(data_dir, n_events=10000, nc_mu: float = 10.0, np_mu: float = 1.0, p_s: float = 5.0, include_noise: bool = True) -> None:
        logging.info(f'Generating data at {data_dir}.')
        events = ToyCaloDataModuleMixin._generate_events(n_events, nc_mu, np_mu, p_s, include_noise)
        for i in tqdm(range(n_events)):
            df = events[i]
            event_path = data_dir / f'{i:05}.pkl'
            df.to_pickle(event_path)

    def make_data(self):
        self.generate(self.data_dir, nc_mu=self.nc_mu, np_mu=self.np_mu, p_s=self.p_s, include_noise=self.include_noise)

    def data_exists(self) -> bool:
        return len(set(self.data_dir.glob('*'))) != 0

if __name__ == '__main__':
    from pathlib import Path
    data_dir = Path('/data/toy_calo')
    ToyCaloDataModuleMixin.generate(data_dir, n_events=10000)
