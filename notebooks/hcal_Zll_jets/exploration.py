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
dm = HCalZllJetsOffsetDataModule.from_config()
# %%
dataset = dm.train_dataloader().dataset
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
dataset._get_df(1)
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
