# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.experiments.hcal_experiment import HCalExperiment
from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy

# %%
exp = HCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))


# %%
cs = ClusteringStudy(exp)


# %%
cs.pq(use_weights=True)


# %%
cs.resolution(nevents=50, splits=('val',))