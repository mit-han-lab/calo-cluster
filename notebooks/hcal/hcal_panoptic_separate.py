# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.experiments.hcal_experiment import HCalExperiment
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift

# %%
exp = HCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))


# %%
ps = PanopticStudy(exp)


# %%
ps.confusion_matrix()

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_label=0)
ps.pq(nevents=1, use_weights=True, ignore_semantic_labels=(0,))

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=False)
ps.pq(use_weights=True)

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_label=0)
ps.qualitative_cluster_study(n=1)

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_label=0)
ps.resolution(nevents=50, splits=('val',))
# %%
