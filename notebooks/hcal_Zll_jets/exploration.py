# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsOffsetDataModule
import pandas as pd
import plotly.express as px
from calo_cluster.evaluation.utils import get_palette
import scipy
import numpy as np
import math
from tqdm import tqdm
# %%
dm = HCalZllJetsOffsetDataModule.from_config(overrides=['dataset.min_eta=-1.5', 'dataset.max_eta=1.5'])
# %%
dm.setup('fit')
# %%
dataset = dm.train_dataloader().dataset
# %%
def _generate_events(n_events):
    evts = []
    for i in tqdm(range(n_events)):
        n_clusters = scipy.stats.poisson.rvs(mu=10)
        cluster_energies = scipy.stats.expon.rvs(size=n_clusters)
        cluster_phi = scipy.stats.uniform.rvs(scale=2*math.pi, loc=-math.pi, size=n_clusters)
        cluster_eta = scipy.stats.arcsine.rvs(scale=3, loc=-1.5, size=n_clusters)
        cluster_r = []
        cluster_npoints = []
        for j in range(n_clusters):
            npoints = 0
            while npoints < 1:
                npoints = scipy.stats.poisson.rvs(mu=cluster_energies[j])
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
            p_x = scipy.stats.norm.rvs(scale=5, size=npoints)
            p_y = scipy.stats.norm.rvs(scale=5, size=npoints)
            p_z = scipy.stats.norm.rvs(scale=5, size=npoints)
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
    return evts
# %%
toy_evts = _generate_events(100)
toy_p_list = []
toy_c_list = []
for i in range(100):
    particles = toy_evts[i]
    mask = particles['semantic_id'] == 1
    particles = particles[mask]
    clusters = dm.get_clusters(particles, cluster_col='instance_id')
    toy_p_list.append(particles)
    toy_c_list.append(clusters)

toy_flat_c = pd.concat(toy_c_list)
toy_flat_p = pd.concat(toy_p_list)

# %%
hcal_p_list = []
hcal_c_list = []
for i in range(100):
    particles = dataset._get_df(i)
    mask = particles['pf_hit'] != 1
    particles = particles[mask]
    clusters = dm.get_clusters(particles, cluster_col='PFcluster0Id')
    hcal_p_list.append(particles)
    hcal_c_list.append(clusters)
hcal_flat_c = pd.concat(hcal_c_list)
hcal_flat_p = pd.concat(hcal_p_list)
# %%
def make_cluster_hists(flat_clusters):
    px.histogram(flat_clusters, x='r').show()
    px.histogram(flat_clusters, x='nconstituents').show()
    px.histogram(flat_clusters, x='energy', histnorm='probability density').show()
    px.histogram(flat_clusters, x='eta').show()
    px.histogram(flat_clusters, x='phi').show()

# %%
make_cluster_hists(hcal_flat_c)
# %%
make_cluster_hists(toy_flat_c)
# %%
def plot_evt(evt, instance_id):
    plot_df = evt
    plot_df['instance_id'] = evt[instance_id].astype(str)
    return px.scatter_3d(plot_df, x='x', y='y', z='z', color='instance_id', color_discrete_sequence=get_palette(plot_df['instance_id']))

def plot_angular(evt, instance_id):
        plot_df = evt
        plot_df['instance_id'] = evt[instance_id].astype(str)
        return px.scatter(plot_df, x='eta', y='phi', color='instance_id', color_discrete_sequence=get_palette(plot_df['instance_id']))
# %%
for i in range(10):
    fig = plot_angular(toy_evts[i], instance_id='instance_id')
    fig.show()
# %%
for i in range(10):
    fig = plot_angular(hcal_p_list[i], instance_id='PFcluster0Id')
    fig.show()


# %%
px.histogram(hcal_flat_p, 'phi')
# %%
hcal_flat_p
# %%
make_cluster_hists

# %%
px.histogram(scipy.stats.expon.rvs(scale=0.04, size=10000))
# %%
