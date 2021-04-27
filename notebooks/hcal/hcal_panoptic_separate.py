# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from hgcal_dev.evaluation.studies.clustering_study import ClusteringStudy
from hgcal_dev.evaluation.experiments.antikt_hcal_experiment import AntiKtHCalEvent, AntiKtHCalExperiment
from hgcal_dev.evaluation.experiments.hcal_experiment import HCalExperiment
from hgcal_dev.evaluation.studies.panoptic_study import PanopticStudy
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.clustering.identity_clusterer import IdentityClusterer
from hgcal_dev.datasets.hcal import HCalDataModule
import pandas as pd
# %%
exp = HCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))


# %%
ps = PanopticStudy(exp)


# %%
ps.confusion_matrix()

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_class_label=0)
ps.pq(nevents=100, use_weights=True, ignore_class_labels=(0,))

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=False)
ps.pq(use_weights=True)

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_class_label=0)
ps.qualitative_cluster_study(n=3)

# %%
ps.clusterer = MeanShift(bandwidth=0.022, use_semantic=True, ignore_class_label=0)
ps.response(nevents=1000, splits=('val',), nbins=10, lo=0.5, hi=5, match_highest=True)

# %%
exp.get_events('val', n=1)[0].input_event.columns
# %%
exp_antikt = AntiKtHCalExperiment(wandb_version=('3ajhmqom', '2vq4h349'))
# %%
ps_antikt = ClusteringStudy(exp_antikt, clusterer=IdentityClusterer())

# %%
exp.get_events('val', n=2)[1].input_event
# %%

val_events = exp.get_events('val', n=10)

clusters = []
for e in val_events:
    clusters.append(HCalDataModule.get_clusters(e))
clusters = pd.concat(clusters)
# %%
