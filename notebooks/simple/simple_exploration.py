# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.clustering.identity_clusterer import IdentityClusterer
from hgcal_dev.evaluation.experiments.simple_experiment import SimpleExperiment
from hgcal_dev.datasets.hcal import HCalDataModule
import pandas as pd
import numpy as np
import plotly.express as px

# %%
exp = SimpleExperiment('rs03d73s', 'epoch=2-step=311.ckpt')
# %%
cs = ClusteringStudy(exp)

# %%
clusterer_factory = lambda bw: MeanShift(bandwidth=bw)
data_dict, figs_dict = cs.bandwidth_study(clusterer_factory=clusterer_factory, nevents=100, stepsize=0.05, bounds=(0.001, 0.15), out_dir='bandwidth_T0.1')

# %%
cs.clusterer = MeanShift(bandwidth=0.022)
cs.qualitative_cluster_study(n=1)

# %%
cs.clusterer = MeanShift(bandwidth=0.022)
cs.pq(nevents=100)

# %%
event = exp.get_events('train', n=1)[0]
# %%
event
# %%
event.input_event
# %%
mat = event.input_event[['x', 'y', 'z']].values
# %%
mat
# %%
import numpy as np
dists = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)
# %%
import plotly.express as px
px.histogram(dists.flatten())
# %%
occs = cs.voxel_occupancy()
# %%
px.histogram(occs)
# %%
occs
# %%
